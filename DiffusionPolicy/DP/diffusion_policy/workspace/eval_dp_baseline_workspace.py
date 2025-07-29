if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
import open3d as o3d
from PIL import Image
import yaml
import time
import math
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.dataset.eval_dp_dataset import eval_dp_dataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from pytorch3d.transforms import axis_angle_to_quaternion, quaternion_to_axis_angle, axis_angle_to_matrix
from pytransform3d.rotations import quaternion_slerp
from diffusion_policy.model.common.normalizer import LinearNormalizer

from e3m5_hand_model import hand_loss, get_e3m5_handmodel
import trimesh
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_euler_angles,
    euler_angles_to_matrix,
    matrix_to_axis_angle,
)
from pytorch3d.loss import chamfer_distance
from e3m5_hand_model import dis_loss, pen_loss


def find_nearest_final_qpose(qpose,target_qpose_pool):
    '''
    param:
        qpose:              [seq_len,30]
        target_qpose_pool   [seq_len,500,30]
    return 
        selected_final_qpose [seq_len,30]
    '''
    wind_len = target_qpose_pool.shape[1]
    seq_len = qpose.shape[0]
    # use euler angle to find the nearest final qpose
    qpose_euler = matrix_to_euler_angles(axis_angle_to_matrix(qpose[...,3:6]),'XYZ')
    target_qpose_pool_euler = matrix_to_euler_angles(axis_angle_to_matrix(target_qpose_pool[...,3:6]),'XYZ')
    qpose_euler_diff = (qpose_euler.unsqueeze(1)[...,:3] - target_qpose_pool_euler[...,:3]).abs().sum(-1)# [seq_len,30]
    # only use trans
    trans_diff = (qpose.unsqueeze(1)[...,:3] - target_qpose_pool[...,:3]).abs().sum(-1)# [seq_len,30]
    top_k = 40
    _, top_k_indices = torch.topk(trans_diff, top_k, dim=1, largest=False)  # [seq_len,5]
    batch_indices = torch.arange(seq_len).unsqueeze(1).expand(-1, top_k)  # [seq_len,5]
    filtered_trans_diff = qpose_euler_diff[batch_indices, top_k_indices]  # [seq_len,5]
    _, min_local_indices = torch.min(filtered_trans_diff, dim=1)  # [seq_len]
    final_indices = top_k_indices[torch.arange(seq_len), min_local_indices]  # [seq_len]
    selected_final_qpose = target_qpose_pool[torch.arange(seq_len), final_indices, :]
    return selected_final_qpose,final_indices



def slerp_quat(v1_,v2_,inter_num):
    '''
    param:
        v1_rot: [4,] torch.tensor
        v2_rot: [4,] torch.tensor
        inter_num: int
    return 
        slerp: [inter_num + 2,4]
    '''
    device = v1_.device
    v1 = v1_.cpu().numpy()
    v2 = v2_.cpu().numpy()
    inter_num += 2
    slerp = np.zeros((inter_num, 4), dtype=np.float32)
    
    interpolation_weight =np.linspace(0, 1, inter_num)
    for i, weight in enumerate(interpolation_weight):
        slerp[i] = quaternion_slerp(v1, v2, weight,shortest_path = True)
    return torch.from_numpy(slerp).to(device=device, dtype=v1_.dtype)


def interpolate_motion(v1, v2, _num_steps=20):
    '''
    param:
        v1,v2 need to
        because interpolation must be applied on quat,so the axis angle rotation will be changed to quat
    return:
        [_num_steps,qpos_dim]
    '''
    if v1.dim() == 1:   
        v1 = v1.unsqueeze(0)
    if v2.dim() == 1:
        v2 = v2.unsqueeze(0)
    interpolate_motion = torch.zeros(_num_steps + 2,v1.shape[-1]).to(v1.device)
    v1_quat = axis_angle_to_quaternion(v1[...,3:6]).squeeze(0)
    v2_quat = axis_angle_to_quaternion(v2[...,3:6]).squeeze(0)
    interpolate_motion[...,3:6] = quaternion_to_axis_angle(slerp_quat(v1_quat,v2_quat,_num_steps))
    weights = torch.linspace(0, 1, _num_steps + 2, dtype=torch.float).view(-1, 1).to("cuda" if torch.cuda.is_available() else "cpu")
    interpolate_motion[...,:3] = (1 - weights) * v1[...,:3] + weights * v2[...,:3]    
    interpolate_motion[...,6:] = (1 - weights) * v1[...,6:] + weights * v2[...,6:]    
    return interpolate_motion


def compute_pose_distance(pose1, pose2):
    trans_dist = (pose1[:3] - pose2[:3]).norm() * 100  # m -> cm
    R1 = axis_angle_to_matrix(pose1[3:6].view(1,3)).view(3,3)
    R2 = axis_angle_to_matrix(pose2[3:6].view(1,3)).view(3,3)
    R_diff = R1 @ R2.transpose(-2, -1)
    w_trans, w_rot = 1.0, 1.0
    return w_trans * trans_dist

def create_folders(base_folder, sub_folders):
    """
    Create specified directory and its subdirectories, and return paths of all subdirectories.
    Args:
        base_folder (str): Base directory path.
        sub_folders (list): List of subdirectories to create under base directory.
    Returns:
        list: List of complete paths for all created subdirectories.
    """
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    folder_paths = {}
    for folder in sub_folders:
        folder_path = os.path.join(base_folder, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        folder_paths[folder] = folder_path
    return folder_paths

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionUnetHybridWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']
    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # configure model
        self.model: DiffusionUnetHybridImagePolicy = hydra.utils.instantiate(cfg.policy)
        self.ema_model: DiffusionUnetHybridImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)
        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())
        # configure training state
        self.global_step = 0
        self.epoch = 0
        self.device = cfg.training.device
        self.n_action_steps = cfg.n_action_steps
        self.n_obs_steps = cfg.n_obs_steps
        self.horizon = cfg.horizon
        self.hand_model = get_e3m5_handmodel(1, self.device)
        
        
    def preprocess_batch(self, batch):
        """
        Simplify batch preprocessing by handling common operations.
        """
        # Move batch to device
        batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
        # Extract and process data
        obs_dict = batch['obs']
        processed_data = {
            "agent_pos": obs_dict["agent_pos"].squeeze(0),  # [1, 16, 28] -> [16, 28]
            "gt_actions": batch['action'].squeeze(0),  # [1, 16, 28] -> [16, 28]
            "objpcd_intact": batch["objpcd_intact"].permute(0, 1, 3, 2).squeeze(0),  # [16, 4096, 3]
            "objpcd_normal_intact": batch["objpcd_normal_intact"].permute(0, 1, 3, 2).squeeze(0),  # [16, 500, 3]
            "final_grasp_group": batch["final_grasp_group"].squeeze(0),  # [16, 500, 28] = [horizon, final_grasp_num, 28]
            "final_grasp": batch["final_grasp"].squeeze(0),
            "velocity": batch["velocity"].squeeze(0),
            "traj_index": batch["traj_index"][0, 0],  # [1, 16] -> float
            "obs_objpcd": batch["obs_objpcd"].squeeze(0)  # [16, 3, 4096]
        }
        return processed_data

    def insert_zeros(self, qpose):
        '''
        param:
            qpose: [seq_len, 28]
        return
            qpose_30: [seq_len, 30]
        '''
        # Create a zeros tensor to insert
        num_zeros = 2
        zeros_actions = torch.zeros(
            qpose.shape[0], 
            num_zeros
        ).to(self.device)
        # Concatenate along the last dimension at the specified position
        insert_position = 6
        updated_actions = torch.cat(
            [qpose[:, :insert_position], zeros_actions, qpose[:, insert_position:]], 
            dim=1
        )
        return updated_actions
    
    def insert_zeros_group(self, qpose):
        '''
        param:
            qpose: [n, seq_len, 28]
        return
            qpose_30: [n, seq_len, 30]
        '''
        # Create a zeros tensor to insert
        num_zeros = 2
        zeros_actions = torch.zeros(
            qpose.shape[0], 
            qpose.shape[1], 
            num_zeros
        ).to(self.device)
        # Concatenate along the last dimension at the specified position
        insert_position = 6
        updated_actions = torch.cat(
            [qpose[:,:, :insert_position], zeros_actions, qpose[:, :,insert_position:]], 
            dim=2
        )
        return updated_actions

    def remove_zeros(self, qpose):
        '''
        param:
            qpose: [seq_len, 30]
        return:
            qpose_28: [seq_len, 28]
        '''
        # Remove the 6th and 7th indices (indexing from 0)
        updated_actions = torch.cat([qpose[:, :6], qpose[:, 8:]], dim=1)
        return updated_actions
    
    def update_obs_dict(self, obs_dict, agent_pos_list):
        """
        Update specific keys in the observation dictionary with stacked tensor values.
        Parameters:
            obs_dict (dict): Original observation dictionary.
        Returns:
            dict: Updated observation dictionary.
        """
        #import pdb; pdb.set_trace()
        new_obs_dict = {}
        for key, value in obs_dict.items():
            new_obs_dict[key] = value  # Copy the original tensor
            if key == "agent_pos":
                new_obs_dict[key][:, :self.n_obs_steps, :] = torch.stack(agent_pos_list, dim=1)
        return new_obs_dict

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        # resume training
        cfg.training.resume = True
        device = cfg.training.device
        if cfg.training.resume:
            lastest_ckpt_path = cfg.checkpoint_path
            print(f"Resuming from checkpoint {lastest_ckpt_path}")
            self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = eval_dp_dataset(cfg.task.dataset.zarr_path, cfg.task.dataset.traj_num, cfg.task.dataset.train_mask_start, 
                                                 cfg.task.dataset.train_mask_end, cfg.task.dataset.val_mask_start, cfg.task.dataset.val_mask_end,
                                                 pad_before=1, pad_after=7, horizon=self.horizon)
        assert isinstance(dataset, BaseImageDataset)
        # TODO: load the whole dataset normalizer
        normalizer = LinearNormalizer()
        normalizer.params_dict = torch.load(cfg.task.normalizer.path)
        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        print("len(dataset) = ", len(dataset))
        print("len(val_dataset) = ", len(val_dataset))

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)
        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)
        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )
        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        # save batch for sampling
        train_sampling_batch = None
        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
        
        # initialize the global variables: (related to config)
        qpose_dim = 28
        traj_num = 0 # the current traj number
        last_traj_index = 1 # used to determine whether the traj has changed to another
        Is_traj_end = False # used to determine whether a traj has ended: reach interpolation
        Is_interpolate = False # used to determine whether is interpolating
        Interpolate_begin = False
        inter_begin_pose = torch.zeros((1, qpose_dim)).to(self.device)
        eval_folder = cfg.eval_folder # the visualize folder of this ckpt
        if not os.path.exists(eval_folder):
            os.makedirs(eval_folder)
        traj_folder = os.path.join(eval_folder, f"traj_{traj_num}") # the traj folder
        sub_folder_names = ["pred_action", "gt_action", "final_grasp", "object", "itp_action", "itp_gt_action", 
                            "itp_final_grasp", "itp_object"]
        sub_folder_paths = create_folders(traj_folder, sub_folder_names) # a directory
        Is_visualize = cfg.visualize # whether to save .ply files
        n_obs_steps = cfg.n_obs_steps # the num of obs steps the model takes in
        n_action_steps = cfg.n_action_steps # the num of action steps the model outputs
        
        # evaluation hyperparameter
        gt_total_pener_frame = cfg.gt_total_pener_frame
        gt_max_pener_depth = cfg.gt_max_pener_depth
        total_infer_frame_threshold = cfg.total_infer_frame_threshold
        interpolation_threshold = cfg.interpolation_threshold
        
        # evaluation metrics:
        metrics = {
            "total_infer_frame": 0,
            "total_infer_time": 0,
            "once_infer_time": 0,
            "traj_accum_length": 0,
            "max_pener_depth": 0,
            "total_pener_frame": 0,
            "dp_safety_rate": 0,
            "dp_success_rate": 0,
            "itp_infer_frame": 0
        }
        # visualized list: pred_hand and itp_hand are put in different lists, because dp infer hand and itp_hand have different colors
        # visualize uniformly after each traj inference is completed
        visualizations = {
            "pred_action": [],
            "gt_action": [],
            "final_grasp": [],
            "object": [],
            "itp_action": [],
            "itp_gt_action": [],
            "itp_final_grasp": [],
            "itp_object": []
        }
        
        # begin evaluating
        with torch.no_grad():
            # with tqdm.tqdm(val_dataloader_iter, desc=f"Validation epoch {self.epoch}", leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
            with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    # get the obs and determine whether the traj has changed
                    processed_data = self.preprocess_batch(batch)
                    traj_index = processed_data["traj_index"]
                    ## TODO: 2/20 change to resume eval, need to change back!
                    agent_pos = processed_data["agent_pos"]
                    gt_actions = processed_data["gt_actions"]
                    objpcd_intact = processed_data["objpcd_intact"]
                    objpcd_normal_intact = processed_data["objpcd_normal_intact"]
                    final_grasp = processed_data["final_grasp"]
                    final_grasp_group = processed_data["final_grasp_group"]
                    obs_objpcd = processed_data["obs_objpcd"]
                    # Initialize obs for the first trajectory in validation:
                    if metrics["total_infer_frame"] == 0:
                        agent_pos_list = [agent_pos[i,:].unsqueeze(0) for i in range(self.n_obs_steps)]
                        last_traj_index = traj_index   
                        last_agent_trans = agent_pos_list[self.n_obs_steps-1][0,:3]
                        
                    # The batch for this trajectory has ended, we can start calculating metrics
                    if traj_index != last_traj_index:
                        metrics["total_pener_frame"] = max(0, metrics["total_pener_frame"]-gt_total_pener_frame)
                        metrics["max_pener_depth"] = max(0, metrics["max_pener_depth"]-gt_max_pener_depth*(metrics["total_infer_frame"]-metrics["itp_infer_frame"]))
                        metrics["dp_safety_rate"] = 0 if metrics["total_pener_frame"] > 0 else 1
                        metrics["dp_success_rate"] = 1 if Is_interpolate else 0
                        wandb.log({
                            "total_infer_frame": metrics["total_infer_frame"],
                            "inference_once_time": metrics["total_infer_time"] / metrics["total_infer_frame"],
                            "inference_total_time": metrics["total_infer_time"],
                            "traj_accum_length": metrics["traj_accum_length"],
                            "max_pener_depth": metrics["max_pener_depth"] / metrics["total_infer_frame"],
                            "total_pener_frame": metrics["total_pener_frame"],
                            "dp_safety_rate": metrics["dp_safety_rate"],
                            "dp_success_rate": metrics["dp_success_rate"],
                            "itp_infer_rate": metrics["itp_infer_frame"] / metrics["total_infer_frame"] 
                        })
                        # Start visualizing all .ply files for this traj, all items in list are trimesh type
                        if Is_visualize:
                            len_dp_data = len(visualizations["pred_action"])
                            len_itp_data = len(visualizations["itp_action"])
                            ## TODO: 2/23 测试运行时间
                            for i in range(len_dp_data):
                                pred_hand_mesh = visualizations["pred_action"][i] # qpose: [1, 30]
                                gt_hand_mesh = visualizations["gt_action"][i]  # qpose: [1, 30]
                                final_grasp_mesh = visualizations["final_grasp"][i]  # qpose: [1, 30]
                                object_mesh = visualizations["object"][i]  # object_mesh: [4096, 3] 
                            
                                np.save(f'{sub_folder_paths["pred_action"]}/pred_hand_{i}.npy', pred_hand_mesh.cpu().numpy())
                                np.save(f'{sub_folder_paths["gt_action"]}/gt_action_{i}.npy', gt_hand_mesh.cpu().numpy())
                                np.save(f'{sub_folder_paths["final_grasp"]}/final_grasp_{i}.npy', final_grasp_mesh.cpu().numpy())
                                object_mesh.export(f'{sub_folder_paths["object"]}/object_{i}.ply')
                                
                            for i in range(len_itp_data):
                                pred_hand_mesh = visualizations["itp_action"][i] # qpose: [1, 30]
                                gt_hand_mesh = visualizations["itp_gt_action"][i] # qpose: [1, 30]
                                final_grasp_mesh = visualizations["itp_final_grasp"][i] # qpose: [1, 30]
                                object_mesh = visualizations["itp_object"][i] # object_mesh: [4096, 3]
                                
                                np.save(f'{sub_folder_paths["itp_action"]}/itp_hand_{i}.npy', pred_hand_mesh.cpu().numpy())
                                np.save(f'{sub_folder_paths["itp_gt_action"]}/itp_gt_hand_{i}.npy', gt_hand_mesh.cpu().numpy())
                                np.save(f'{sub_folder_paths["itp_final_grasp"]}/itp_final_grasp_{i}.npy', final_grasp_mesh.cpu().numpy())
                                object_mesh.export(f'{sub_folder_paths["itp_object"]}/itp_object_{i}.ply')
                        traj_num += 1
                        traj_folder = f"{eval_folder}/traj_{traj_num}"
                        sub_folder_paths = create_folders(traj_folder, sub_folder_names)
                        Is_traj_end = False
                        Is_interpolate = False
                        Interpolate_begin = False
                        agent_pos_list = [agent_pos[i,:].unsqueeze(0) for i in range(self.n_obs_steps)]
                        last_traj_index = traj_index
                        last_agent_trans = agent_pos_list[self.n_obs_steps-1][0,:3]
                        metrics = {
                            "total_infer_frame": 0,
                            "total_infer_time": 0,
                            "once_infer_time": 0,
                            "traj_accum_length": 0,
                            "max_pener_depth": 0,
                            "total_pener_frame": 0,
                            "dp_safety_rate": 0,
                            "dp_success_rate": 0,
                            "itp_infer_frame": 0
                        }
                        visualizations = {
                            "pred_action": [],
                            "gt_action": [],
                            "final_grasp": [],
                            "object": [],
                            "itp_action": [],
                            "itp_gt_action": [],
                            "itp_final_grasp": [],
                            "itp_object": []
                        }                       
                        
                    # There are still batches containing information about this trajectory
                    elif traj_index == last_traj_index:
                        # This trajectory ended early because the hand successfully grasped the object -> good; skip remaining batches until traj_index != last_traj_index
                        if Is_traj_end:
                            last_traj_index = traj_index
                            continue
                        # The traj is not over yet, continue infer; the way to infer depends on the Is_interpolate variable!
                        else:
                            if Interpolate_begin:
                                curr_obs_index = self.n_obs_steps - 1
                                curr_hand_pose = self.insert_zeros(inter_begin_pose) # [1, 28]->[1, 30] the last hand pose from previous dp inference, which is the n_obs_steps-th element in the newly updated agent_pos_list
                                curr_objpcd_intact = objpcd_intact[curr_obs_index] # [4096, 3]
                                curr_objpcd_normal_intact = objpcd_normal_intact[curr_obs_index] # [4096, 3]
                                curr_gt_hand_pose = self.insert_zeros(gt_actions[curr_obs_index].unsqueeze(0)) # [28] -> [1, 28] -> [1, 30]
                                curr_final_grasp_group = self.insert_zeros(final_grasp_group[curr_obs_index]) # [500, 28] = [final_grasp_num, 28] -> [500, 30]
                                # Select and return a final_grasp based on curr_hand_pose and curr_final_grasp_group
                                curr_final_grasp, _ = find_nearest_final_qpose(curr_hand_pose, curr_final_grasp_group.unsqueeze(0)) # [1, 30]
                                # Compute the distance between curr_hand_pos and curr_final_grasp, then round down to get num_interpolate_steps
                                itp_num_steps = math.floor(compute_pose_distance(curr_hand_pose.squeeze(0), curr_final_grasp.squeeze(0)) / 2)
                                # If itp_num_steps == 0, then is_traj_end = True
                                if itp_num_steps == 0:
                                    Is_traj_end = True
                                # Interpolate between curr_hand_pose and final_grasp, return the first interpolation
                                interpolate_hand = interpolate_motion(curr_hand_pose, curr_final_grasp, itp_num_steps)[1].unsqueeze(0) # [1, 30] because index=0 is the starting hand pose
                                # Save the first interpolation hand mesh to itp_action list
                                visualizations["itp_action"].append(interpolate_hand)
                                # Save curr_gt_hand_pose mesh into itp_gt_hand list
                                visualizations["itp_gt_action"].append(curr_gt_hand_pose)
                                # Save curr_final_grasp mesh into itp_final_grasp list
                                visualizations["itp_final_grasp"].append(curr_final_grasp)
                                # Save objpcd_intact mesh into itp_object list
                                curr_objpcd_intact_points = curr_objpcd_intact.cpu().numpy()
                                curr_objpcd_intact_mesh = trimesh.points.PointCloud(curr_objpcd_intact_points)
                                visualizations["itp_object"].append(curr_objpcd_intact_mesh)
                                # Compute the distance between interpolation hand and final_grasp_hand, then update traj_accum_length in metrics
                                metrics["traj_accum_length"] += torch.sqrt(torch.sum((curr_hand_pose[0][:3] - interpolate_hand[0][:3])**2))
                                # Update itp_infer_frame in metrics
                                metrics["itp_infer_frame"] += 1
                                # Update the starting position for the next interpolation
                                inter_begin_pose = self.remove_zeros(interpolate_hand)
                            if metrics["total_infer_frame"] % self.n_action_steps == 0 and Interpolate_begin == False:
                                if Is_interpolate:
                                    # Only consider the first pos of each batch, because each batch only moves one obs forward
                                    curr_obs_index = self.n_obs_steps - 1
                                    curr_hand_pose = self.insert_zeros(inter_begin_pose) # [1, 28]->[1, 30] the last hand pose from previous dp inference, which is the n_obs_steps-th element in the newly updated agent_pos_list
                                    curr_objpcd_intact = objpcd_intact[curr_obs_index] # [4096, 3]
                                    curr_objpcd_normal_intact = objpcd_normal_intact[curr_obs_index] # [4096, 3]
                                    curr_gt_hand_pose = self.insert_zeros(gt_actions[curr_obs_index].unsqueeze(0)) # [28] -> [1, 28] -> [1, 30]
                                    curr_final_grasp_group = self.insert_zeros(final_grasp_group[curr_obs_index]) # [16, 500, 28] -> [500, 28] = [final_grasp_num, 28] -> [500, 30]
                                    # Select and return a final_grasp based on curr_hand_pose and curr_final_grasp_group
                                    curr_final_grasp, _ = find_nearest_final_qpose(curr_hand_pose, curr_final_grasp_group.unsqueeze(0)) # [1, 30]
                                    # Compute the distance between curr_hand_pos and curr_final_grasp, then round down to get num_interpolate_steps
                                    itp_num_steps = math.floor(compute_pose_distance(curr_hand_pose.squeeze(0), curr_final_grasp.squeeze(0)) / 2)
                                    
                                    if itp_num_steps == 0:
                                        Is_traj_end = True
                                    # Interpolate between curr_hand_pose and final_grasp, return the first interpolation
                                    interpolate_hand = interpolate_motion(curr_hand_pose, curr_final_grasp, itp_num_steps)[1].unsqueeze(0) # [1, 30] 因为index=0是起点的hand pose
                                    # Save the first interpolation hand mesh to itp_action list
                                    visualizations["itp_action"].append(interpolate_hand)
                                    # Save curr_gt_hand_pose mesh into itp_gt_hand list
                                    visualizations["itp_gt_action"].append(curr_gt_hand_pose)
                                    # Save curr_final_grasp mesh into itp_final_grasp list
                                    visualizations["itp_final_grasp"].append(curr_final_grasp)
                                    # Save objpcd_intact mesh into itp_object list
                                    curr_objpcd_intact_points = curr_objpcd_intact.cpu().numpy()
                                    curr_objpcd_intact_mesh = trimesh.points.PointCloud(curr_objpcd_intact_points)
                                    visualizations["itp_object"].append(curr_objpcd_intact_mesh)
                                    # Compute the distance between interpolation hand and final_grasp_hand, then update traj_accum_length in metrics
                                    metrics["traj_accum_length"] += torch.sqrt(torch.sum((curr_hand_pose[0][:3] - interpolate_hand[0][:3])**2))
                                    # Update itp_infer_frame in metrics
                                    metrics["itp_infer_frame"] += 1
                                    # Set Interpolate_begin to True
                                    Interpolate_begin = True
                                    # Update the starting position for the next interpolation
                                    inter_begin_pose = self.remove_zeros(interpolate_hand)
                                else:
                                    # Update obs
                                    new_obs_dict = self.update_obs_dict(batch["obs"], agent_pos_list)
                                    # Compute model infer time, then update total_infer_time and once_infer_time in metrics
                                    start_time = time.time()
                                    pred_actions = self.model.predict_action(new_obs_dict)["action"].squeeze(0) # [b, n_action_steps, 28] = [1, n_action_steps, 28] -> [n_action_steps, 28]
                                    end_time = time.time()
                                    metrics["total_infer_time"] += (end_time - start_time)
                                    # Find the nearest final_grasp for each infer_hand, the index of final_grasp starts from self.n_obs_steps
                                    obs_start_index = self.n_obs_steps
                                    pred_actions_30 = self.insert_zeros(pred_actions) # [n_action_steps, 28] -> [n_action_steps, 30]
                                    chosen_final_grasp_30, _ = find_nearest_final_qpose(pred_actions_30, self.insert_zeros_group(final_grasp_group[obs_start_index:obs_start_index+self.n_action_steps])) # [n_action_steps, 30]
                                    
                                    for i in range(self.n_action_steps):
                                        curr_agent_trans = pred_actions[i][:3]
                                        metrics["traj_accum_length"] += torch.sqrt(torch.sum((curr_agent_trans - last_agent_trans)**2))
                                        last_agent_trans = curr_agent_trans
                                        pred_hand_pcd = self.hand_model.get_surface_points(q=pred_actions_30[i].unsqueeze(0)).to(dtype=torch.float32)
                                        curr_objpcd_normal_intact = objpcd_normal_intact[obs_start_index+i]
                                        curr_objpcd_intact = objpcd_intact[obs_start_index+i]
                                        curr_obj_pcd_nor = torch.cat((curr_objpcd_intact.unsqueeze(0), curr_objpcd_normal_intact.unsqueeze(0)), dim=-1).to(dtype=torch.float32)
                                        pen_loss_value = pen_loss(curr_obj_pcd_nor, pred_hand_pcd)
                                        metrics["max_pener_depth"] += pen_loss_value
                                        metrics["total_pener_frame"] += (1 if pen_loss_value > 0 else 0)
                                        # Save pred_hand, gt_hand, objpcd_intact and final_grasp_mesh into corresponding lists
                                        curr_gt_hand_pose = self.insert_zeros(gt_actions[obs_start_index+i].unsqueeze(0))
                                        curr_final_grasp = chosen_final_grasp_30[i].unsqueeze(0)
                                        curr_objpcd_intact_points = curr_objpcd_intact.cpu().numpy()
                                        curr_objpcd_intact_mesh = trimesh.points.PointCloud(curr_objpcd_intact_points)
                                        # change saving format from mesh to qpose
                                        visualizations["pred_action"].append(pred_actions_30[i].unsqueeze(0))
                                        visualizations["gt_action"].append(curr_gt_hand_pose)
                                        visualizations["final_grasp"].append(curr_final_grasp)
                                        visualizations["object"].append(curr_objpcd_intact_mesh)
                                        
                                    last_pred_hand = pred_actions_30[-1]
                                    last_final_grasp = chosen_final_grasp_30[-1]
                                    distance = compute_pose_distance(last_pred_hand, last_final_grasp)
                                    cd_loss_value, _ = chamfer_distance(torch.tensor(objpcd_intact[obs_start_index+self.n_action_steps-1], dtype=torch.float).unsqueeze(0), torch.tensor(objpcd_intact[obs_start_index+self.n_action_steps-2], dtype=torch.float).unsqueeze(0), point_reduction='mean', batch_reduction='mean')
                                    if distance < interpolation_threshold and cd_loss_value != 0 and metrics["total_infer_frame"] > total_infer_frame_threshold:
                                        Is_interpolate = True
                                        inter_begin_pose = pred_actions[-1].unsqueeze(0) # [1, 28]
                                        print("success")
                                    # If > 10cm, then continue dp infer
                                    # Update agent_pos_list and final_grasp_list as the obs for the next infer
                                    if self.n_action_steps >= self.n_obs_steps:
                                        agent_pos_list = [pred_actions[j].unsqueeze(0) for j in range(self.n_action_steps-self.n_obs_steps, self.n_action_steps)]
                                    else:
                                        keep_old_obs = self.n_obs_steps - self.n_action_steps
                                        agent_pos_list = agent_pos_list[-keep_old_obs:]
                                        agent_pos_list.extend([pred_actions[j].unsqueeze(0) for j in range(self.n_action_steps)])
                                            
                    # Update total_infer_frame in metrics and last_traj_index
                    metrics["total_infer_frame"] += 1
                    last_traj_index = traj_index
            self.global_step += 1
            self.epoch += 1    
            

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetHybridWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()                                   
                        

                                
                            
                        
                    

        
        
       
            