from typing import Any, Tuple, Dict
import os
import sys
sys.path.append(os.getcwd())
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig, OmegaConf
from datasets.misc import  collate_fn_general
from datasets.base import DATASET
import json
from utils.registry import Registry
from utils.utils import safe_search_file, turn_hand_full_qpose_6d_to_axis_angle
from pytorch3d.transforms import axis_angle_to_matrix,matrix_to_rotation_6d
from os.path import join as pjoin
from tqdm import tqdm 
from utils.e3m5_hand_model import get_e3m5_handmodel
import trimesh as tm
from typing import DefaultDict
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import yaml

def load_grasp_begin_index(grasp_data_path:str):
    yaml_file_path = pjoin(grasp_data_path,"motion_info.yaml")
    with open(yaml_file_path,"r") as f:
        grasp_begin_index = int(yaml.load(f,Loader=yaml.FullLoader)["grasp_begin_frame_index"])
    return grasp_begin_index

def load_from_json(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    train_split = {"sub_names":data["train_sub_names"],"obj_names":data["train_obj"]}
    val_split = {"sub_names":data["val_sub_names"],"obj_names":data["val_obj"]}
    test_split = {"sub_names":data["test_sub_names"],"obj_names":data["test_obj"]}
    all_split = {"sub_names": data["all_sub_names"],"obj_names":data["all_obj"]}
    return train_split, test_split, val_split, all_split

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def hand_qpose_axis_angle_to_rot_6d(hand_qpose):
    hand_rot_6d = matrix_to_rotation_6d(axis_angle_to_matrix(hand_qpose[...,3:6]))
    hand_qpose = torch.cat([hand_qpose[...,:3],hand_rot_6d,hand_qpose[...,6:]],dim = -1)
    return hand_qpose



def load_seq_data_path(dataset_dir):
    data_dir_path_save_path = "assets/file_path_list/motion_file_save_path_list.pt" # make sure every time the data load is same
    if os.path.exists(data_dir_path_save_path):
        all_grasp_data_dir_path = torch.load(data_dir_path_save_path)
    else:
        all_grasp_data_dir_path = safe_search_file(dataset_dir, "qpos.pt")
        os.makedirs(os.path.dirname(data_dir_path_save_path),exist_ok=True)
        torch.save(all_grasp_data_dir_path,data_dir_path_save_path)
    return all_grasp_data_dir_path

@DATASET.register()
class DynamicMotion(Dataset):
    # read json
    input_file = "./../dataset/dataset_split.json"
    _train_split, _test_split, _val_split, _all_split = load_from_json(input_file)

    def __init__(self , cfg: DictConfig, phase: str,sub_path_list:list = None, **kwargs: Dict) -> None:
        super(DynamicMotion, self).__init__()
        self.phase = phase
        self.data_type = torch.float32
        if self.phase == 'train':
            print("uses train")
            self.split = self._train_split
        elif self.phase == 'test':
            print("uses test")
            self.split = self._test_split
        elif self.phase == 'all':
            self.split = self._all_split
        else:
            raise Exception('Unsupported phase.')
        self.use_mano_filter_dexgraspanything_collision_final_pose = cfg.use_mano_filter_dexgraspanything_collision_final_pose
        self.use_mano_filter_cvae_collision_final_pose = cfg.use_mano_filter_cvae_collision_final_pose
        self.use_dexgraspanything_final_pose = cfg.use_dexgraspanything_final_pose
        self.use_cvae_final_pose = cfg.use_cvae_final_pose

        self.device = cfg.device
        self.original_points_num = cfg.original_points_num
        self.sample_points = cfg.sample_points
        self.past_frames = cfg.past_frames
        self.future_frames = cfg.future_frames
        self.use_mesh_model_surface_pcd = cfg.use_mesh_model_surface_pcd
        self.load_kinect_pcd = cfg.load_kinect_pcd
        # Initialize data structures
        self.window_size = self.past_frames + self.future_frames
        
        self.downsample_slice = slice(None,None,cfg.downsample_num)
        self.use_grasp_seq = cfg.use_grasp_seq
        self.load_final_qpose = cfg.load_final_qpose

        ## resource folders
        self.dataset_dir = os.path.join('./../dataset/DexH2R_dataset')
        self.sub_path_list = sub_path_list
        ## load data
        self._pre_load_data()
        
        self.seq_flag = -1

    def jud_path_in_split(self,grasp_data_dir_path:str):
        '''
        different with the grasp jud
        '''

        sub_name, obj_name = self.split_sub_obj_name(grasp_data_dir_path)
        sub_name = int(sub_name)
        union_test_val_sub_name = self._test_split["sub_names"] + self._val_split["sub_names"]
        union_test_val_obj_name = self._test_split["obj_names"] + self._val_split["obj_names"]


        if self.phase == "train":
            if sub_name in self.split["sub_names"] and obj_name in self.split["obj_names"]:
                return True
        elif self.phase == "test":
            if (sub_name in self._train_split["sub_names"] and obj_name in self._test_split["obj_names"]) or \
                (sub_name in self._test_split["sub_names"] and obj_name in self._train_split["obj_names"]) or \
                (sub_name in union_test_val_sub_name and obj_name in union_test_val_obj_name):
                return True
        elif self.phase == 'all':
            return True
        return False
    
    def load_item(self,dir_path,data_name):
        data_path = pjoin(dir_path,data_name)
        data = torch.load(data_path).to(self.device).to(torch.float)
        return data
    

    def load_mesh_model(self):
        self.mesh_model = {}
        self.mesh_surface_pcd = {}
        mesh_model_dir_path = "./../dataset/object_model/"
        mesh_surface_pcd_dir_path = f"./../dataset/obj_surface_pcd_{self.original_points_num}/"

        pcd_resample_indices = torch.randperm(self.original_points_num)[:self.sample_points].to(self.device)
        for file_name in os.listdir(mesh_model_dir_path):
            if file_name.endswith(".obj"):
                model_path = pjoin(mesh_model_dir_path,file_name)
                model_name = file_name[:-4]
                self.mesh_model[model_name] = tm.load(model_path)

        for file_name in os.listdir(mesh_surface_pcd_dir_path):
            model_path = pjoin(mesh_surface_pcd_dir_path,file_name)
            model_name = file_name[:-4]
            surface_pcd = torch.from_numpy(np.array(tm.load(model_path).vertices)).to(torch.float)[...,pcd_resample_indices,:3].contiguous()
            self.mesh_surface_pcd[model_name] = torch.cat([surface_pcd,torch.ones(surface_pcd.shape[0],1)],dim = -1)

    def split_sub_obj_name(self,dir_path:str):
        split_list = dir_path.split("/")
        sub_name = split_list[-3]
        obj_name = split_list[-2]
        return sub_name,obj_name

    def load_one_seq_data(self,grasp_data_dir_path:str):
        '''
        every seq need to be unfold seperately, so different seq won't be mixed
        unfold function, will let the window size be the final dim
        param:
            grasp_data_dir_path: .../ is a dir(it contains the qpos.pt)
        '''
        grasp_begin_index = load_grasp_begin_index(grasp_data_dir_path)
        if self.use_grasp_seq:
            data_slice = slice(0,None)
        else:
            data_slice = slice(0,grasp_begin_index)

        sub_name,obj_name = self.split_sub_obj_name(grasp_data_dir_path)

        one_seq_gt_qpose = self.load_item(grasp_data_dir_path,"qpos.pt")
        one_seq_gt_qpose = hand_qpose_axis_angle_to_rot_6d(one_seq_gt_qpose)[data_slice][self.downsample_slice].contiguous().to(self.data_type)

        # load final qpose
        if self.use_mano_filter_cvae_collision_final_pose:
            final_qpose_path = pjoin(grasp_data_dir_path,"model_final_qpose_cvae_mano_filter.pt")
        elif self.use_mano_filter_dexgraspanything_collision_final_pose:
            final_qpose_path = pjoin(grasp_data_dir_path,"model_final_qpose_cvae_dexgraspanything_mano_filter.pt")
        elif self.use_dexgraspanything_final_pose:
            final_qpose_path = pjoin(grasp_data_dir_path,"model_final_qpose_cvae_dexgraspanything.pt")
        elif self.use_cvae_final_pose:
            final_qpose_path = pjoin(grasp_data_dir_path,"model_final_qpose_cvae.pt")

        final_qpose_pool = torch.load(final_qpose_path).to(self.device)
        final_qpose_pool = hand_qpose_axis_angle_to_rot_6d(final_qpose_pool)[data_slice][self.downsample_slice].contiguous().to(self.data_type)

        # load obj pose
        obj_pose = self.load_item(grasp_data_dir_path,"obj_pose.pt")[data_slice][self.downsample_slice].contiguous().to(self.data_type)
        seq_obj_pcd_xyz = None
        if self.load_kinect_pcd:
            seq_obj_pcd = self.load_item(grasp_data_dir_path,"real_obj_pcd_xyz.pt")
            seq_obj_pcd_xyz = seq_obj_pcd[...,self.pcd_resample_indices,:3][data_slice][self.downsample_slice].contiguous().to(self.data_type)

        seq_length = one_seq_gt_qpose.shape[0]
        num_windows = seq_length - self.window_size + 1
        return sub_name, obj_name, one_seq_gt_qpose, final_qpose_pool, obj_pose, num_windows, seq_obj_pcd_xyz

    def _pre_load_data(self) -> None:
        """ Load dataset    
        because this is the motion data, so, the transform don't need to be transfer to the object
        but all the transform need to be 6d, so the effect would be better
        must test, the world frame in the motion mid or in the fixed ground, which one is better
        """

        self.load_mesh_model()
        self.data_indices = []
        
        self.scene_pcds = []
        self.qpose = []
        self.all_seq_final_qpose = []
        self.obj_pose = []
        self.window_info = DefaultDict(dict)
        self.mesh_model = {}

        self.sub_name_list = []
        self.obj_name_list = []

        all_grasp_data_dir_path = load_seq_data_path(self.dataset_dir)

        all_grasp_data_dir_path = [str(item_path.parent) for item_path in all_grasp_data_dir_path]
        self.split_grasp_data_path = [item_path for item_path in all_grasp_data_dir_path if  self.jud_path_in_split(item_path)]

        # self.split_grasp_data_path = self.split_grasp_data_path[:20]

        if self.sub_path_list is not None:
            self.split_grasp_data_path = self.sub_path_list
            # self.split_grasp_data_path = [item_path for item_path in self.split_grasp_data_path if item_path in self.sub_path_list]

        # in original goal paper,they use past 5 frames to predict future 10 frames
        self.window_size = self.past_frames + self.future_frames
        self.pcd_resample_indices = torch.randperm(self.original_points_num)[:self.sample_points].to(self.device)

        # for seq_idx,grasp_data_dir_path in tqdm(enumerate(self.split_grasp_data_path), desc="loading data",total=len(self.split_grasp_data_path)):
        #     '''
        #     every seq need to be unfold seperately, so different seq won't be mixed
        #     unfold function, will let the window size be the final dim
        #     '''
        #     sub_name,obj_name = self.split_sub_obj_name(grasp_data_dir_path)
        #     self.sub_name_list.append(sub_name)
        #     self.obj_name_list.append(obj_name)


        #     one_seq_gt_qpose = self.load_item(grasp_data_dir_path,"qpos.pt")
        #     one_seq_gt_qpose = hand_qpose_axis_angle_to_rot_6d(one_seq_gt_qpose)[self.downsample_slice].contiguous()
        #     self.qpose.append(one_seq_gt_qpose)

        #     # load final qpose
        #     if self.use_mano_filter_cvae_collision_final_pose:
        #         final_qpose_path = pjoin(grasp_data_dir_path,"final_qpose_mano_filter_completion.pt")
        #     elif self.use_mano_filter_dexgraspanything_collision_final_pose:
        #         final_qpose_path = pjoin(grasp_data_dir_path,"final_qpose_using_dexgraspanything_mano_filter_completion.pt")
        #     elif self.use_zym_final_pose:
        #         final_qpose_path = pjoin(grasp_data_dir_path,"final_qpose_using_dexgraspanything.pt")
        #     else:
        #         final_qpose_path = pjoin(grasp_data_dir_path,"final_qpose.npy")


        #     final_qpose_pool = torch.load(final_qpose_path).to(self.device)
        #     final_qpose_pool = hand_qpose_axis_angle_to_rot_6d(final_qpose_pool)[self.downsample_slice].contiguous()
        #     self.all_seq_final_qpose.append(final_qpose_pool)

        #     # load obj pose
        #     obj_pose = self.load_item(grasp_data_dir_path,"obj_pose.pt")[self.downsample_slice].contiguous()
        #     self.obj_pose.append(obj_pose)
        #     if self.load_kinect_pcd:
        #         seq_obj_pcd = self.load_item(grasp_data_dir_path,"real_obj_pcd_xyz.pt")
        #         seq_obj_pcd_xyz = seq_obj_pcd[...,self.pcd_resample_indices,:3][self.downsample_slice].contiguous()
        #         self.scene_pcds.append(seq_obj_pcd_xyz)



        #     seq_length = one_seq_gt_qpose.shape[0]
        #     num_windows = seq_length - self.window_size + 1
        #     for w in range(num_windows):
        #         self.window_info[window_idx + w] = {
        #             'seq_idx': seq_idx,
        #             'start_frame': w
        #         }
        #     window_idx += num_windows

        with ThreadPoolExecutor(max_workers=12) as executor:  # 可以根据系统调整worker数量
            all_motion_data_list = list(tqdm(executor.map(self.load_one_seq_data, self.split_grasp_data_path), total=len(self.split_grasp_data_path), desc="Loading motion data"))

        self.sub_name_list = [motion_data[0] for motion_data in all_motion_data_list]
        self.obj_name_list = [motion_data[1] for motion_data in all_motion_data_list]
        self.qpose = [motion_data[2] for motion_data in all_motion_data_list]
        self.all_seq_final_qpose = [motion_data[3] for motion_data in all_motion_data_list]
        self.obj_pose = [motion_data[4] for motion_data in all_motion_data_list]
        all_window_num_list = [motion_data[5] for motion_data in all_motion_data_list]
        if self.load_kinect_pcd:
            self.scene_pcds = [motion_data[6] for motion_data in all_motion_data_list]
        
        self.window_info = defaultdict(dict)
        window_idx = 0
        for seq_idx, num_windows in enumerate(all_window_num_list):
            for w in range(num_windows):
                self.window_info[window_idx + w] = {
                    'seq_idx': seq_idx,
                    'start_frame': w
                }
            window_idx += num_windows
        
        self.all_data_len = window_idx
        
    def get_full_seq_data(self,seq_idx):

        seq_obj_pose = self.obj_pose[seq_idx]

        # if self.use_mesh_model_surface_pcd:
        #     seq_pcd = (self.mesh_surface_pcd[self.obj_name_list[seq_idx]].unsqueeze(0) @ seq_obj_pose.permute(0,2,1))[...,:3]
        # else:

        seq_qpose = self.qpose[seq_idx]
        seq_final_qpose_pool = self.all_seq_final_qpose[seq_idx]
        
        obj_meshmodel_surface_pcd = (self.mesh_surface_pcd[self.obj_name_list[seq_idx]].unsqueeze(0) @ seq_obj_pose.permute(0,2,1))[...,:3]

        if self.use_mesh_model_surface_pcd:
            seq_pcd = obj_meshmodel_surface_pcd
        else:   
            seq_pcd = self.scene_pcds[seq_idx]

        seq_data = {
            'seq_pcd':seq_pcd,
            'seq_qpose':seq_qpose,
            'seq_final_qpose_pool':seq_final_qpose_pool,
            'obj_meshodel_surface_pcd':obj_meshmodel_surface_pcd,
            'sub_name':self.sub_name_list[seq_idx],
            'obj_name':self.obj_name_list[seq_idx],
            'gtdata_dir_path':self.split_grasp_data_path[seq_idx]
        }
        # print(seq_data["sub_name"])
        # print(seq_data["obj_name"])
        return seq_data
    def get_seq_len(self):
        return len(self.qpose)
    
    def __len__(self):
        return self.all_data_len
    
    def __getitem__(self, index) -> Tuple:
        
        window_info = self.window_info[index]
        seq_idx = window_info['seq_idx']
        start_frame = window_info['start_frame']
        window_interval = slice(start_frame, start_frame + self.window_size)
        wind_gt_hand_qpose = self.qpose[seq_idx][window_interval]
        wind_final_qpose = self.all_seq_final_qpose[seq_idx][window_interval]

        data = {
            'wind_hand_qpose': wind_gt_hand_qpose, 
            # 'wind_obj_pcd': wind_obj_pcd,
            # 'final_qpose_pool': wind_final_qpose,
            # 'obj_surface_pcd': wind_obj_meshmodel_surface_pcd
        }
        if self.load_final_qpose:
            data["final_qpose_pool"] = wind_final_qpose
        
        if self.use_mesh_model_surface_pcd:
            wind_obj_pose = self.obj_pose[seq_idx][window_interval]
            wind_obj_meshmodel_surface_pcd = (self.mesh_surface_pcd[self.obj_name_list[seq_idx]].unsqueeze(0) @ wind_obj_pose.permute(0,2,1))[...,:3]
            wind_obj_pcd = wind_obj_meshmodel_surface_pcd
        else:

            wind_obj_pcd = self.scene_pcds[seq_idx][window_interval]

        


        data["wind_obj_pcd"] = wind_obj_pcd[:self.past_frames]
        return data
    
    def get_dataloader(self, **kwargs):
        return DataLoader(self, **kwargs)





if __name__ == '__main__':

    config_path = "./configs/task/motion_net_gen.yaml"
    cfg = OmegaConf.load(config_path)
    # dataset = DynamicMotion_no_cat(cfg.dataset, 'all')
    dataloader = DynamicMotion(cfg.dataset, 'train').get_dataloader(collate_fn=collate_fn_general,
                                                                                  batch_size = 1,
                                                                                  num_workers=1,
                                                                                  pin_memory=True,
                                                                                  shuffle=False)
    
    save_dir_path = "./test_meshes/test_dynamic_grasp_dataset/motion_net"
    hand_save_dir_path = pjoin(save_dir_path,"hand")
    obj_pcd_save_dir_path = pjoin(save_dir_path,"obj_pcd")
    final_qpose_save_dir_path = pjoin(save_dir_path,"final_qpose")
    os.makedirs(hand_save_dir_path,exist_ok=True)
    os.makedirs(obj_pcd_save_dir_path,exist_ok=True)
    os.makedirs(final_qpose_save_dir_path,exist_ok=True)
    from utils.e3m5_hand_model import get_e3m5_handmodel
    hand_model = get_e3m5_handmodel("cuda",more_surface_points=False)

    # TODO need to attention, the dataloader return is [batchsize,window_size,one_data]
    import open3d as o3d
    save_data_index = 0
    save_hand_index = 0
    
    for index, data in tqdm(enumerate(dataloader),desc="saving data"):# the batchsize is 1
        for key in data:
            if torch.is_tensor(data[key]):
                data[key] = data[key].to("cuda")
        pcd_data = data["wind_obj_pcd"].squeeze(0) # [1,wind,pcd_num,3]
        # normal_data = data["normal"].squeeze(        hand_data = data["wind_hand_qpose"].squeeze(0) # [1,wind,20,qpose_dim]
        final_qpose_pool = data["final_qpose_pool"].squeeze(0) # [1,wind,20,qpose_dim]
        hand_data = turn_hand_full_qpose_6d_to_axis_angle(data['wind_hand_qpose'].squeeze(0))

        for data_index in torch.arange(0,pcd_data.shape[0]): 


            one_pcd_data = pcd_data[data_index,...].squeeze()
            # one_normal_data = normal_data[4,...].squeeze()
            # one_hand_data = hand_data[4,...].squeeze()
            # one_future_data = data["future_frames_hand_points"].squeeze()[5]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(one_pcd_data.cpu().numpy().reshape((-1,3)))
            # pcd.normals = o3d.utility.Vector3dVector(one_normal_data.cpu().numpy().reshape((-1,3)))
            o3d.io.write_point_cloud(pjoin(obj_pcd_save_dir_path,f"seq_{index}_pcd_{data_index}.ply"),pcd)
        
        # for hand_data_index in np.arange(hand_data.shape[0]):
        #     hand_mesh = hand_model.get_meshes_from_q(hand_data[hand_data_index].unsqueeze(0))
        #     hand_mesh.export(pjoin(hand_save_dir_path,f"seq_{index}_hand_{hand_data_index}.ply"))

        
        for hand_data_index in np.arange(final_qpose_pool.shape[1]):
            hand_mesh = hand_model.get_meshes_from_q(final_qpose_pool[4,hand_data_index].unsqueeze(0))
            hand_mesh.export(pjoin(final_qpose_save_dir_path,f"seq_{index}_final_hand_{hand_data_index}.ply"))

    #     # save_data_index += 1

    #     # for future_index,future_frame_hand_pcd in enumerate(data["future_frames_hand_points"].squeeze()):
    #     # pcd = o3d.geometry.PointCloud()
    #     # pcd.points = o3d.utility.dVector3dVector(future_frames_hand_points.cpu().numpy().reshape((-1,3)))
    #     # o3d.io.write_point_cloud(pjoin(obj_pcd_save_dir_path,f"future_hand_{index}.ply"),pcd)

    #     # pcd = o3d.geometry.PointCloud()
    #     # pcd.points = o3d.utility.Vector3dVector(data["final_hand_points"].squeeze().cpu().numpy().reshape((-1,3)))
    #     # o3d.io.write_point_cloud(pjoin(obj_pcd_save_dir_path,f"final_hand_{index}.ply"),pcd)
    # save_dir_path = ./test_meshes/test_motion_pose"
    # import colorama
    # colorama.init(autoreset=True)
    # from chamfer_distance import ChamferDistance as chamfer_dist

    # device = "cuda"
    # chd = chamfer_dist()
    # wrong_seq_list = []
    # for data_index in tqdm(torch.arange(dataset.get_seq_len())):
    #     seq_data = dataset.get_full_seq_data(data_index)
    #     mesh_model_pcd = seq_data['obj_meshodel_surface_pcd'].to(device) # [seq_len,num_pcd,3]
    #     obs_raw_pcd = seq_data['seq_pcd'].to(device)

    #     # import trimesh as tm    
    #     # rawpcd = tm.PointCloud(vertices=obs_raw_pcd[0].cpu().numpy())
    #     # pcd = tm.PointCloud(vertices=mesh_model_pcd[0].cpu().numpy())

    #     # pcd.export(pjoin(save_dir_path,f"model_pcd_{data_index}.ply"))
    #     # rawpcd.export(pjoin(save_dir_path,f"raw_pcd_{data_index}.ply"))
    #     dist1, dist2, idx1, idx2 = chd(mesh_model_pcd,obs_raw_pcd)# [seq_len, num_pcd]

    #     pcd_num = dist1.shape[1]
    #     num_pcd_points_away_to_hand_surface_points_lower_than_four_cm = (dist1 < 0.01).sum(dim = 1)  # [seq_len]
    #     contact_qpose_mask = num_pcd_points_away_to_hand_surface_points_lower_than_four_cm < 2000 # remained content is < 400 then should filter
        
    #     if any(contact_qpose_mask):
    #         print(colorama.Fore.RED + seq_data["sub_name"] + " " + seq_data["obj_name"])
    #         # wrong_seq_list.append(seq_data["sub_name"] + " " + seq_data["obj_name"])

    #     # wrong_seq_save_file_path = "./wrong_list/wrong_seq.txt"
    #     # with open(wrong_seq_save_file_path, 'w', encoding='utf-8') as f:
    #     #     for string in wrong_seq_list:
    #     #         f.write(str(string) + '\n')

