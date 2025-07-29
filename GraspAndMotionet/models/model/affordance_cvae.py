import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from models.model.pointnet_encoder import PointNetEncoder
from models.model.CVAE import VAE
from models.base import MODEL
import os
# from bps_torch.bps import bps_torch
from omegaconf import DictConfig
from typing import Dict, Tuple
import trimesh as tm
from copy import deepcopy
from utils.e3m5_hand_model import get_e3m5_handmodel, hand_loss
import time
from os.path import join as pjoin
from tqdm import tqdm   
import open3d as o3d


@MODEL.register()
class affordance_cvae(nn.Module):
    _joint_angle_lower = torch.tensor([-0.5235988, -0.7853982, -0.43633232, 0., 0., 0., -0.43633232, 0., 0., 0.,
                                       -0.43633232, 0., 0., 0., 0., -0.43633232, 0., 0., 0., -1.047, 0., -0.2618,
                                       -0.5237, 0.])
    _joint_angle_upper = torch.tensor([0.17453292, 0.61086524, 0.43633232, 1.5707964, 1.5707964, 1.5707964, 0.43633232,
                                       1.5707964, 1.5707964, 1.5707964, 0.43633232, 1.5707964, 1.5707964, 1.5707964,
                                       0.6981317, 0.43633232, 1.5707964,  1.5707964, 1.5707964, 1.047, 1.309, 0.2618,
                                       0.5237, 1.])
    #The estimated maximum and minimum joint rotation angles.


    # dexgraspnet norm, because our data train on dexgraspnet first,so we must align to it
    e3m5_orginal_trans = torch.tensor([ 0.0000, -0.0100,  0.2470])
    
    _global_trans_lower = torch.tensor([-0.13128923, -0.10665303, -0.45753425]) + e3m5_orginal_trans
    _global_trans_upper = torch.tensor([0.12772022, 0.22954416, -0.21764427]) + e3m5_orginal_trans


    _NORMALIZE_LOWER = -1.
    _NORMALIZE_UPPER = 1.

    def angle_denormalize(self, joint_angle: torch.Tensor):
        joint_angle_denorm = joint_angle + (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        joint_angle_denorm /= (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER)
        joint_angle_denorm = joint_angle_denorm * (self._joint_angle_upper - self._joint_angle_lower) + self._joint_angle_lower
        return joint_angle_denorm
    
    def trans_denormalize(self, global_trans: torch.Tensor):
        global_trans_denorm = global_trans + (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        global_trans_denorm /= (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER)
        global_trans_denorm = global_trans_denorm * (self._global_trans_upper - self._global_trans_lower) + self._global_trans_lower
        return global_trans_denorm


    def __init__(self, 
                 cfg: DictConfig,
                 train_flag = True,
                 *args,
                 **kwargs):
        super(affordance_cvae, self).__init__()
        


        # self.demo_save_index = 0

        self.train_flag = train_flag

        self.device = cfg.device #"cuda" if torch.cuda.is_available() else "cpu"
        self.hand_loss = hand_loss(_device = self.device,
                                   dis_loss_weight = cfg.dis_loss_weight,
                                   pen_loss_weight = cfg.pen_loss_weight,
                                   hand_pose_loss_weight = cfg.hand_pose_loss_weight,
                                   hand_surface_point_mse_loss_weight = cfg.hand_surface_point_mse_loss_weight)


        self.dataset_name = cfg.dataset_name
        self.norm = cfg.norm

        self.obj_inchannel = 3 # the xyz, the normals don't need to be encoded
        self.hand_param_dim = cfg.hand_param_dim
        self.cvae_encoder_sizes = list(cfg.encoder_layer_sizes)
        self.cvae_latent_size = cfg.cvae_latent_size
        self.cvae_decoder_sizes = list(cfg.decoder_layer_sizes)

        self.batch_size = cfg.batch_size

        self.hand_encoder = PointNetEncoder(channel=3)
        self.hand_encoder_dim = 1024
        self.obj_encoder = PointNetEncoder(channel=3)
        self.cvae_condition_size = 1024
        
        # encode end 
        self.cvae_encoder_sizes[0] = self.hand_encoder_dim # the first dim match the encode size of the hand
        self.cvae = VAE(encoder_layer_sizes=self.cvae_encoder_sizes,
                        latent_size=self.cvae_latent_size,
                        decoder_layer_sizes=self.cvae_decoder_sizes,
                        condition_size=self.cvae_condition_size)
        
        self.hand_model = get_e3m5_handmodel(device = self.device)
    
        self._joint_angle_lower = self._joint_angle_lower.to(self.device)
        self._joint_angle_upper = self._joint_angle_upper.to(self.device)
        self._global_trans_lower = self._global_trans_lower.to(self.device)
        self._global_trans_upper = self._global_trans_upper.to(self.device)

    def encode_data(self,data,training_flag = True):
        hand_glb_feature = None
        obj_glb_feature = None
        
        if training_flag:
            hand_glb_feature = self.hand_encoder(data["hand_surface_points"]) # [B, num,dim]
        obj_glb_feature = self.obj_encoder(data["obj_pcd"].float()) # res[B, 1024]

        return hand_glb_feature, obj_glb_feature
    
    

    def forward(self, 
                data: Dict,
                ):
        '''
        :param obj_pc: [B, 3+n, N1]
        :param hand_param: [B, 27]
        :return: reconstructed hand vertex
        '''

        # # debug
        # output_dir_path = "test_meshes/debug"
        # original_hand_qpos_translation = self.trans_denormalize(data["hand_qpos"][:,:3])
        # original_hand_qpos_angle = self.angle_denormalize(data["hand_qpos"][:,3:])
        # original_hand_qpos = torch.cat([original_hand_qpos_translation,original_hand_qpos_angle],dim = 1)
        # original_hand_qpos[:,3:5] = 0
        # norm_hand_save_dir_path = pjoin(output_dir_path,"norm_hand")
        # os.makedirs(norm_hand_save_dir_path,exist_ok=True)
        # for hand_index in tqdm(torch.arange(original_hand_qpos.shape[0]),desc="norm_hand"):
            
        #     hand_mesh = self.hand_model.get_meshes_from_q(original_hand_qpos[hand_index:hand_index+1])
        #     hand_mesh.export(pjoin(norm_hand_save_dir_path,f"{hand_index}.ply"))
        

        # origin_hand_save_dir_path = pjoin(output_dir_path,"origin_hand")
        # os.makedirs(origin_hand_save_dir_path,exist_ok=True)
        # for hand_index in tqdm(torch.arange(data["original_hand_qpos"].shape[0]),desc="origin_hand"):
        #     hand_mesh = self.hand_model.get_meshes_from_q(data["original_hand_qpos"][hand_index:hand_index+1])
        #     hand_mesh.export(pjoin(origin_hand_save_dir_path,f"{hand_index}.ply"))
    
        # hand_surface_pcd_save_dir_path = pjoin(output_dir_path,"hand_surface_pcd")
        # os.makedirs(hand_surface_pcd_save_dir_path,exist_ok=True)
        # for hand_index in tqdm(torch.arange(data["hand_surface_points"].shape[0]),desc="hand_surface_pcd"):
        #     hand_pcd = tm.PointCloud(data["hand_surface_points"][hand_index].cpu().numpy())
        #     hand_pcd.export(pjoin(hand_surface_pcd_save_dir_path,f"{hand_index}.ply"))

        # obj_pcd_save_dir_path = pjoin(output_dir_path,"obj_pcd")
        # os.makedirs(obj_pcd_save_dir_path,exist_ok=True)
        # for obj_index in tqdm(torch.arange(data["obj_pcd"].shape[0]),desc="obj_pcd"):
        #     obj_pcd = o3d.geometry.PointCloud()
        #     obj_pcd.points = o3d.utility.Vector3dVector(data["obj_pcd"][obj_index].cpu().numpy())
        #     obj_pcd.normals = o3d.utility.Vector3dVector(data["obj_pcd_normal"][obj_index].cpu().numpy())
        #     o3d.io.write_point_cloud(pjoin(obj_pcd_save_dir_path,f"{obj_index}.ply"), obj_pcd)
        # # debug end

        # exit(0)


        hand_glb_feature, obj_glb_feature = self.encode_data(data)# the input hand qpose is 27 dim,but the model output is 25 dim
        recon_x, means, log_var, z = self.cvae(hand_glb_feature, obj_glb_feature) # recon_x: [B, 778*3]

        recon_x = torch.cat([recon_x[:,:3],torch.zeros(recon_x.shape[0],2).to(recon_x.device),recon_x[:, 3:]],dim = 1) # remove the wrist
        recon_x = recon_x.contiguous().view(self.batch_size, 27)

        if self.norm:
            recon_x[:, :3] = self.trans_denormalize(recon_x[:, :3])
            recon_x[:, 3:] = self.angle_denormalize(recon_x[:, 3:])

        recon_x[:, 3:5] = 0 # denorm may let wrist not 0

        cvae_loss_data = {}
        cvae_loss_data["mean"] = means
        cvae_loss_data["log_var"] = log_var
        hand_loss = self.hand_loss.cal_loss(data,recon_x,
                                            cvae_loss_data=cvae_loss_data)

        return hand_loss 
    

    
    def sample(self,data: Dict, k_sample: int=1,need_time = False):
        self.batch_size = k_sample

        # inference
        # TODO add parameter mse
        start_infer_time = time.time()
        _, obj_glb_feature = self.encode_data(data,training_flag = False)
        # print(data["pos"].shape)
        recon_x = self.cvae.inference(self.batch_size, obj_glb_feature)

        recon_x = torch.cat([recon_x[:,:3],torch.zeros(recon_x.shape[0],2).to(recon_x.device),recon_x[:, 3:]],dim = 1)
        end_infer_time = time.time()
        # print("-"*100)
        du_time = end_infer_time - start_infer_time
        # print(f"infer {k_sample} has {du_time}")
        # print("-"*100)
        recon_x = recon_x.contiguous().view(self.batch_size, 27)

        if self.norm:
            recon_x[:, :3] = self.trans_denormalize(recon_x[:, :3])
            recon_x[:, 3:] = self.angle_denormalize(recon_x[:, 3:])

        recon_x[:, 3:5] = 0
        
        if need_time:
            return recon_x,data,du_time
        else:
            return recon_x,data
        

