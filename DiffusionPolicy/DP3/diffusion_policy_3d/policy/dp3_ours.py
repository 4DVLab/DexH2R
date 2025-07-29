from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint
import copy
import time
import pytorch3d.ops as torch3d_ops
from diffusion_policy_3d.env_runner.e3m5_hand_model import hand_loss
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy_3d.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.model_util import print_params
from diffusion_policy_3d.model.vision.pointnet_extractor import DP3Encoder

class DP3_ours(BasePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            loss_weight=None,
            # parameters passed to step
            **kwargs):
        super().__init__()
        self.loss_weight = loss_weight
        self.condition_type = condition_type

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
            
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])


        obs_encoder = DP3Encoder(observation_space=obs_dict,
                                                   img_crop_shape=crop_shape,
                                                out_channel=encoder_output_dim,
                                                pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                                use_pc_color=use_pc_color,
                                                pointnet_type=pointnet_type,
                                                )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            if "cross_attention" in self.condition_type:
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * n_obs_steps
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] pointnet_type: {self.pointnet_type}", "yellow")



        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        
        
        self.noise_scheduler_pc = copy.deepcopy(noise_scheduler)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps


        print_params(self)
        
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            condition_data_pc=None, condition_mask_pc=None,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler


        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)


        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]


            model_output = model(sample=trajectory,
                                timestep=t, 
                                local_cond=local_cond, global_cond=global_cond)
            
            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, ).prev_sample
            
                
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]   


        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        # this_n_point_cloud = nobs['imagin_robot'][..., :3] # only use coordinate
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        this_n_point_cloud = nobs['point_cloud']
        
        
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        # get action
        start = To 
        end = start + self.n_action_steps
        if end <= self.horizon:
            action = action_pred[:,start:end]
        else:
            action = action_pred[:,start:]
        
        
        # get prediction


        result = {
            'action': action,
            'action_pred': action_pred,
        }
        
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input

        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        
       
        
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)

            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)
            # this_n_point_cloud = this_nobs['imagin_robot'].reshape(batch_size,-1, *this_nobs['imagin_robot'].shape[1:])
            this_n_point_cloud = this_nobs['point_cloud'].reshape(batch_size,-1, *this_nobs['point_cloud'].shape[1:])
            this_n_point_cloud = this_n_point_cloud[..., :3]
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()


        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        


        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        
        pred = self.model(sample=noisy_trajectory, 
                        timestep=timesteps, 
                            local_cond=local_cond, 
                            global_cond=global_cond)


        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        elif pred_type == 'v_prediction':
            # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # https://github.com/huggingface/diffusers/blob/v0.11.1-patch/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # sigma = self.noise_scheduler.sigmas[timesteps]
            # alpha_t, sigma_t = self.noise_scheduler._sigma_to_alpha_sigma_t(sigma)
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t, sigma_t = self.noise_scheduler.alpha_t[timesteps], self.noise_scheduler.sigma_t[timesteps]
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * trajectory
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")
        
        new_batch_size = batch_size* self.horizon
        hand_loss_instance = hand_loss(
            _batch_size=new_batch_size,
            _device=self.device,
            loss_type="l2",
            use_spen_loss=True if self.loss_weight['spen_coeff'] > 0 else False,
            use_dis_loss=True if self.loss_weight['dis_coeff'] > 0 else False,
            use_pen_loss=True if self.loss_weight['pen_coeff'] > 0 else False,
            use_hand_pose_loss = True if self.loss_weight['hand_pose_loss_coeff'] > 0 else False,
            # newly-added: change to False
            hand_norm = False,
            use_final_grasp_mse_loss = True if self.loss_weight['final_grasp_coeff'] > 0 else False,
        )
        # 这里还和hand loss的设置不太一样，不过可以尝试将batch_size * n_action_steps这样作为new batch size，就可以一起求loss了
        data_dict = {}
        # gt_action: b, h, c -> b*h, c
        data_dict["x"] = batch["action"][:, :self.horizon, :].clone().reshape(new_batch_size, -1).to(self.device)
        # objpcd: b, h, n, 3 -> transpose b*h, n, 3
        data_dict["pos"] = batch["objpcd_intact"][:, :self.horizon, :].clone().reshape(new_batch_size, batch["objpcd_intact"].shape[-2], batch["objpcd_intact"].shape[-1]).to(self.device)
        # objpcd_normal: b, h, n, 3 -> transpose b*h, n, 3
        data_dict["normal"] = batch["objpcd_normal_intact"][:, :self.horizon, :].clone().reshape(new_batch_size, batch["objpcd_normal_intact"].shape[-2], batch["objpcd_normal_intact"].shape[-1]).to(self.device)
        # final grasp: b, h, 28 -> b*h, 28
        if 'final_grasp' in batch['obs']:
            data_dict["final_grasp"] = batch['obs']['final_grasp'][:, :self.horizon,:].reshape(new_batch_size, -1).to(self.device)
        else:
            data_dict["final_grasp"] = batch['final_grasp'][:, :self.horizon,:].reshape(new_batch_size, -1).to(self.device)
        # predicted qpose: b, h, c -> b*h, c
        # pred_x0 = torch.tensor(pred_action).reshape(batch_size*self.n_action_steps, -1).to(device)
        # newly-added
        pred_denorm = self.normalizer['action'].unnormalize(pred)
        pred_front = pred_denorm[:,:self.horizon,:]
        pred_x0 = pred_front.clone().reshape(new_batch_size, -1).to(self.device)
        
        #dof 6+22 → 6+2+22
        zeros = torch.zeros(data_dict["x"].shape[0], 2).to(self.device)
        gt_qpose_left = data_dict["x"][:, :6]
        gt_qpose_right = data_dict["x"][:, 6:]
        data_dict["x"] = torch.cat((gt_qpose_left, zeros, gt_qpose_right), dim=1)
        pred_qpose_left = pred_x0[:, :6]
        pred_qpose_right = pred_x0[:, 6:]
        pred_x0 = torch.cat((pred_qpose_left, zeros, pred_qpose_right), dim=1)
        final_grasp_left = data_dict["final_grasp"][:, :6]
        final_grasp_right = data_dict["final_grasp"][:, 6:]
        final_grasp = torch.cat((final_grasp_left, zeros, final_grasp_right), dim=1)
        data_dict['final_grasp'] = final_grasp

        loss_dict = hand_loss_instance.cal_loss(data_dict, pred_x0)
        # 提取各个损失
        dis_loss_value = loss_dict['dis_loss_value']
        pen_loss_value = loss_dict['pen_loss_value']
        spen_loss_value = loss_dict['spen_loss_value']
        loss_main = F.mse_loss(pred, target, reduction='none')
        loss_main = loss_main * loss_mask.type(loss_main.dtype)
        loss_main = reduce(loss_main, 'b ... -> b (...)', 'mean')
        loss_main = loss_main.mean()
        # # 自动调整其他loss的权重系数
        base_loss_main = loss_main.detach()
        auto_scale = base_loss_main / 5.0  # 保持其他loss项比main低一个数量级
        if self.loss_weight['dis_coeff'] > 0:
            dis_coeff = self.loss_weight['dis_coeff'] * auto_scale / (dis_loss_value.detach() + 1e-8)
        else:
            dis_coeff = 0
        if self.loss_weight['pen_coeff'] > 0:
            pen_coeff = self.loss_weight['pen_coeff'] * auto_scale / (pen_loss_value.detach() + 1e-8)
        else:
            pen_coeff = 0
        
        loss = (    self.loss_weight['loss_main_coeff'] * loss_main +
                    dis_coeff * dis_loss_value +
                    pen_coeff * pen_loss_value 
                )
        # loss = (    self.loss_weight['loss_main_coeff'] * loss_main +
        #             self.loss_weight['dis_coeff'] * dis_loss_value +
        #             self.loss_weight['pen_coeff'] * pen_loss_value         
        # )
        loss_dict = {
            'bc_loss': loss.item(),
            'loss_main': loss_main.item()
        }
        if self.loss_weight['spen_coeff'] > 0:
            loss_dict['spen_loss'] = spen_loss_value.item()
        if self.loss_weight['dis_coeff'] > 0:
            loss_dict['dis_loss_value'] = dis_loss_value.item()
        if self.loss_weight['pen_coeff'] > 0:
            loss_dict['pen_loss_value'] = pen_loss_value.item()
        # print('bc_loss', loss.item())
        # print('loss_main', loss_main.item())
        return loss, loss_dict

