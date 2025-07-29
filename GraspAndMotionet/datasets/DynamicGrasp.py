# import debugpy
# debugpy.listen(("localhost", 15000))
# debugpy.wait_for_client()  # wait for debugger to attach


from typing import Any, Tuple, Dict
import os
import sys
sys.path.append(os.getcwd())
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig, OmegaConf
from datasets.misc import collate_fn_general
from datasets.base import DATASET
import json
from utils.registry import Registry
from os.path import join as pjoin
import open3d as o3d
from utils.e3m5_hand_model import get_e3m5_handmodel, pen_loss, dis_loss
from pytorch3d.ops import knn_points
import yaml
from concurrent.futures import ThreadPoolExecutor
from pytorch3d.transforms import axis_angle_to_matrix
from datasets.DynamicMotion import load_seq_data_path, load_grasp_begin_index
from tqdm import tqdm
from chamfer_distance import ChamferDistance as chamfer_dist

def load_from_json(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    train_split = {"sub_names":data["train_sub_names"],"obj_names":data["train_obj"]}
    val_split = {"sub_names":data["val_sub_names"],"obj_names":data["val_obj"]}

    test_split = {"sub_names":data["test_sub_names"] + val_split["sub_names"],"obj_names":data["test_obj"] + val_split["obj_names"]}
    all_split = {"sub_names": data["all_sub_names"],"obj_names":data["all_obj"]}
    return train_split, test_split, all_split


def batch_transfer_hand_rot_to_obj(batch_pcd_xyz,batch_obj_pose,batch_qpose):
    '''
    param:
        batch_pcd_xyz: (N, pcd_num, 3)
        batch_obj_pose: (N, 4, 4)
        batch_qpose: (N, 30)
    return:

    '''
    
    batch_hand_rot_mat = axis_angle_to_matrix(batch_qpose[:,3:6])# [N,3,3]
    batch_hand_trans = batch_qpose[:,0:3]# [N,3]
    
    batch_obj_translation = batch_obj_pose[:,:3,3].clone()#[N,3]

    # process obj pose
    batch_obj_pose[:,:3,3] = 0 # move to origin
    batch_obj_pose[:,:3,:3] = batch_hand_rot_mat.transpose(2,1) @ batch_obj_pose[:,:3,:3]# transfer the hand rot to obj [N,3,3] @ [N,3,3] -> [N,3,3]
    # process obj pose end

    # process pcd
    if batch_pcd_xyz is not None:
        batch_pcd_xyz -= batch_obj_translation.unsqueeze(1)# [N,p,3] - [N,1,3]
        batch_pcd_xyz = (batch_pcd_xyz.unsqueeze(2) @ batch_hand_rot_mat.unsqueeze(1)).squeeze(2)# [N,p,1,3] @ [N,1,3,3] -> [N,p,3]
    # process pcd end

    new_hand_trans = ((batch_hand_trans - batch_obj_translation).unsqueeze(1) @ batch_hand_rot_mat).squeeze(1)# [N,1,3] @ [N,3,3] -> [N,3]

    batch_qpose[:,:3] = new_hand_trans
    batch_qpose[:,3:6] = 0

    return batch_pcd_xyz,batch_obj_pose,batch_qpose




def element_wise_pen_loss(obj_pcd_nor: torch.Tensor, hand_pcd: torch.Tensor):
    """
    Calculate the penalty loss based on point cloud and normal.
    calculate the mean max penetration loss
    :param obj_pcd_nor: B x N_obj x 6 (object point cloud with normals)
    :param hand_pcd: B x N_hand x 3 (hand point cloud)
    :return: pen_loss (scalar)
    """
    b = hand_pcd.shape[0]
    n_obj = obj_pcd_nor.shape[1]
    n_hand = hand_pcd.shape[1]

    # Separate object point cloud and normals
    obj_pcd = obj_pcd_nor[:, :, :3]
    obj_nor = obj_pcd_nor[:, :, 3:6]

    # Compute K-nearest neighbors
    knn_result = knn_points(hand_pcd, obj_pcd, K=1, return_nn=True)
    distances = knn_result.dists
    indices = knn_result.idx
    knn = knn_result.knn
    # distances = distances.sqrt()
    # Extract the closest object points and normals
    hand_obj_points = torch.gather(obj_pcd, 1, indices.expand(-1, -1, 3))
    hand_obj_normals = torch.gather(obj_nor, 1, indices.expand(-1, -1, 3))
    # Compute the signs
    hand_obj_signs = ((hand_obj_points - hand_pcd) * hand_obj_normals).sum(dim=2)
    hand_obj_signs = (hand_obj_signs > 0.).float()
    # Compute collision value
    # collision_value = (hand_obj_signs * hand_obj_dist).mean(dim=1)
    collision_value = (hand_obj_signs * distances.squeeze(2)).max(dim=1).values

    return collision_value


# train cvae not need normalize
# train dex grasp anything need normalize and transfer the base frame
@DATASET.register()
class DynamicGrasp(Dataset):
    """ Dataset for pose generation, training with RealDex Dataset
    """
    
    # read json
    input_file = "./../dataset/dataset_split.json"
    _train_split, _test_split, _all_split = load_from_json(input_file)

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

    def __init__(self , cfg: DictConfig, phase: str,  **kwargs: Dict) -> None:
        super(DynamicGrasp, self).__init__()
        self.phase = phase

        if self.phase == 'train':# need the loading data in the split
            print("uses train")
            self.split = self._train_split
        elif self.phase == 'test':
            print("uses test")
            self.split = self._test_split
        elif self.phase == 'all':
            self.split = self._all_split
        else:
            raise Exception('Unsupported phase.')
        self.device = cfg.device
        self.original_points_num = cfg.original_points_num
        self.sample_points_num = cfg.sample_points_num
        self.use_normal = cfg.use_normal # cal penetration loss
        self.normalize_x = cfg.normalize_x  # x: the qpose
        self.normalize_x_trans = cfg.normalize_x_trans # x: the qpose
        self.use_mesh_model_surface_pcd = cfg.use_mesh_model_surface_pcd # use the mesh model surface pcd

        ## resource folders
        self.dataset_dir = cfg.dataset_dir
        self.obj_mesh_dir_path = "./../dataset/object_model/"
        self.obj_surface_pcd_dir_path = f"./../dataset/obj_surface_pcd_{cfg.original_points_num}/"
        self._joint_angle_lower = self._joint_angle_lower.to(self.device)
        self._joint_angle_upper = self._joint_angle_upper.to(self.device)
        self._global_trans_lower = self._global_trans_lower.to(self.device)
        self._global_trans_upper = self._global_trans_upper.to(self.device)

        ## load data
        self._pre_load_data()
        



    def get_sub_and_obj_name_from_data_path(self,data_path:str):
        path_item = data_path.split("/")
        sub_name = path_item[-4]
        obj_name = path_item[-3]
        return sub_name, obj_name


    def load_one_seq_data(self,grasp_data_path:str):
        '''
        param:
            grasp_data_path: .../qpos.pt

        '''
        grasp_data_dir_path = os.path.dirname(grasp_data_path)
        grasp_begin_index = load_grasp_begin_index(grasp_data_dir_path)
        grasp_data_slice = slice(grasp_begin_index,None)

        grasp_data = torch.load(grasp_data_path)[grasp_data_slice].to(self.device).to(torch.float32)
        obj_pcd = None
        if not self.use_mesh_model_surface_pcd:
            obj_pcd_path = pjoin(grasp_data_dir_path,f"real_obj_pcd_xyz.pt")
            obj_pcd = torch.load(obj_pcd_path)[grasp_data_slice].to(self.device).to(torch.float32)# [batch,pcd_num,3]、
        obj_pose_path = pjoin(grasp_data_dir_path,f"obj_pose.pt")
        obj_pose = torch.load(obj_pose_path)[grasp_data_slice].to(self.device).to(torch.float32)

        one_seq_len = obj_pose.shape[0]
        one_seq_obj_name_list = [grasp_data_path.split("/")[-3]] * one_seq_len
        one_seq_sub_name_list = [grasp_data_path.split("/")[-4]] * one_seq_len


        return grasp_data, obj_pose, obj_pcd, one_seq_obj_name_list, one_seq_sub_name_list 


    def _pre_load_data(self) -> None:
        """ Load dataset

        Args:
            case_only: only load single case for testing, if ture, the dataset will be smaller.
                        This is useful in after-training visul evaluation.
        """
        # TODO
        # split the data
        self.scene_pcds = []
        self.qpose = []
        self.pcd_pose = []
        
        all_grasp_data_path = load_seq_data_path(self.dataset_dir)
        
        split_grasp_data_path = []
        for item_path in all_grasp_data_path:
            item_path = str(item_path)
            sub_name, obj_name = self.get_sub_and_obj_name_from_data_path(item_path)
            if obj_name in self.split["obj_names"]  : #sub_name in self.split["sub_names"]:
                split_grasp_data_path.append(item_path)

        # split_grasp_data_path = [str(item_path) for item_path in all_grasp_data_path if "normal" not in str(item_path) and self.jud_path_in_split(str(item_path))][:100]

        self.sub_name_list = []
        self.obj_name_list = []
        self.obj_pose = []
        self.obj_mesh = {}
        self.obj_surface_pcd = {}
        self.obj_surface_pcd_normals = {}
        self.data_index = []

        model_name_list = [file_name for file_name in os.listdir(self.obj_mesh_dir_path) if file_name.endswith("obj")]
        for model_name in model_name_list:
            model_name_str = model_name.replace(".obj","")
            model_path = pjoin(self.obj_mesh_dir_path,model_name)
            # obj_mesh = o3d.io.read_triangle_mesh(model_path)
            obj_surface_pcd_path = pjoin(self.obj_surface_pcd_dir_path,model_name.replace(".obj",".ply"))
            surface_pcd = o3d.io.read_point_cloud(obj_surface_pcd_path)

            obj_surface_pcd = torch.from_numpy(np.asarray(surface_pcd.points))
            obj_surface_pcd_normals = torch.from_numpy(np.asarray(surface_pcd.normals))
            sample_pcd_indices = torch.randperm(obj_surface_pcd.shape[0])[:self.sample_points_num]
            obj_surface_pcd = obj_surface_pcd[sample_pcd_indices]
            obj_surface_pcd_normals = obj_surface_pcd_normals[sample_pcd_indices]
            
            # self.obj_mesh[model_name_str] = obj_mesh
            self.obj_surface_pcd[model_name_str] = torch.cat([obj_surface_pcd,torch.ones(obj_surface_pcd.shape[0],1)],dim = 1).to(torch.float)# [p,3]
            self.obj_surface_pcd_normals[model_name_str] = obj_surface_pcd_normals.to(torch.float) # [...,3]

        with ThreadPoolExecutor(max_workers=12) as executor:  # 可以根据系统调整worker数量
            all_grasp_data_list = list(tqdm(executor.map(self.load_one_seq_data, split_grasp_data_path), total=len(split_grasp_data_path), desc="Loading grasp data"))
        
        self.qpose = [item[0] for item in all_grasp_data_list]
        self.obj_pose = [item[1] for item in all_grasp_data_list]
        
        self.obj_name_list = [obj_name for item in all_grasp_data_list for obj_name in item[3]]
        self.sub_name_list = [sub_name for item in all_grasp_data_list for sub_name in item[4]]

        self.qpose = torch.cat(self.qpose,dim = 0)
        self.obj_pose = torch.cat(self.obj_pose,dim = 0)
        
        self.scene_pcds = None
        if not self.use_mesh_model_surface_pcd:
            self.scene_pcds = [item[2] for item in all_grasp_data_list]
            self.scene_pcds = torch.cat(self.scene_pcds,dim = 0)[...,:3]

        self.scene_pcds,self.obj_pose,self.qpose = batch_transfer_hand_rot_to_obj(self.scene_pcds,self.obj_pose,self.qpose)
        
        self.obj_pose = self.obj_pose.transpose(1,2)
        self.all_data_len = self.qpose.shape[0]
        
        hand_translation = self.qpose[...,:3]# rotation is 0
        hand_joint_angle = self.qpose[...,6:]
        self.original_qpose = torch.cat([hand_translation,hand_joint_angle],dim = -1)
        
        hand_model = get_e3m5_handmodel("cuda")
        self.hand_surface_points = hand_model.get_surface_points(self.qpose.to("cuda")).to("cpu")

        if self.normalize_x:
            norm_hand_joint_angle = self.angle_normalize(hand_joint_angle)
        if self.normalize_x_trans:
            norm_hand_translation = self.trans_normalize(hand_translation)
        if self.normalize_x and self.normalize_x_trans:
            self.norm_hand_trans_qpose = torch.cat((norm_hand_translation,norm_hand_joint_angle),dim = 1)


        print(f"load data done, DynamicGrasp dataset has {self.all_data_len} items data ")




    def trans_normalize(self, global_trans: torch.Tensor):
        global_trans_norm = torch.div((global_trans - self._global_trans_lower), (self._global_trans_upper - self._global_trans_lower))
        global_trans_norm = global_trans_norm * (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) - (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        return global_trans_norm
    @classmethod
    def trans_denormalize(cls, global_trans: torch.Tensor):
        global_trans_denorm = global_trans + (cls._NORMALIZE_UPPER - cls._NORMALIZE_LOWER) / 2
        global_trans_denorm /= (cls._NORMALIZE_UPPER - cls._NORMALIZE_LOWER)
        global_trans_denorm = global_trans_denorm * (cls._global_trans_upper - cls._global_trans_lower) + cls._global_trans_lower
        return global_trans_denorm

    def angle_normalize(self, joint_angle: torch.Tensor):
        joint_angle_norm = torch.div((joint_angle - self._joint_angle_lower), (self._joint_angle_upper - self._joint_angle_lower))
        joint_angle_norm = joint_angle_norm * (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) - (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        return joint_angle_norm
    @classmethod
    def angle_denormalize(cls, joint_angle: torch.Tensor):
        joint_angle_denorm = joint_angle + (cls._NORMALIZE_UPPER - cls._NORMALIZE_LOWER) / 2
        joint_angle_denorm /= (cls._NORMALIZE_UPPER - cls._NORMALIZE_LOWER)
        joint_angle_denorm = joint_angle_denorm * (cls._joint_angle_upper - cls._joint_angle_lower) + cls._joint_angle_lower
        return joint_angle_denorm

    def __len__(self):
        return self.all_data_len
    
    def __getitem__(self, index: Any) -> Tuple:# all points are 4096 
        
        data = {
            'original_hand_qpos': self.original_qpose[index],  # [global_trans, joint_angle_remove_the_rewrist(22)] 
            "hand_surface_points": self.hand_surface_points[index],
            'sub_name': self.sub_name_list[index],
            'obj_name': self.obj_name_list[index]
        }
        if self.normalize_x_trans: # if norm, the joint angle also be norm
            data["hand_qpos"] = self.norm_hand_trans_qpose[index] 
        else:
            data["hand_qpos"] = self.original_qpose[index]
        if self.use_mesh_model_surface_pcd:
            mesh_model_surface_pcd = (self.obj_surface_pcd[self.obj_name_list[index]] @ self.obj_pose[index])[:,:3] # the obj pose has already been transpose
            data["obj_pcd"] = mesh_model_surface_pcd
        else:# use observation raw pcd
            data["obj_pcd"] = self.scene_pcds[index]

        # data["obj_kinect_pcd"] = self.scene_pcds[index]

        if self.use_normal:
            data["obj_pcd_normal"] = self.obj_surface_pcd_normals[self.obj_name_list[index]] @ self.obj_pose[index][:3,:3]
        return data

    def get_dataloader(self, **kwargs):
        return DataLoader(self, **kwargs)



if __name__ == '__main__':

    # test dataset ouput data
    config_path = "configs/task/grasp_gen_ur.yaml"
    cfg = OmegaConf.load(config_path)
    # print(cfg.dataset)
    dataloader = DynamicGrasp(cfg.dataset, 'test').get_dataloader(batch_size=600,collate_fn=collate_fn_general,num_workers=0, shuffle=False) # pin_memory=True

    device = torch.device('cuda:0')
    hand_model = get_e3m5_handmodel(device = device)
    chd = chamfer_dist()

    save_dir_path = "./test_meshes/DynamicGrasp_dataset"
    os.makedirs(save_dir_path, exist_ok= True)

    obj_pcd_save_dir_path = pjoin(save_dir_path,f"obj_pcd")
    hand_surface_pcd_save_dir_path = pjoin(save_dir_path,f"hand_surface_pcd")
    hand_mesh_save_dir_path = pjoin(save_dir_path,f"hand_mesh")
    orginal_hand_mesh_save_dir_path = pjoin(save_dir_path,f"orginal_hand_mesh")

    os.makedirs(obj_pcd_save_dir_path,exist_ok=True)
    os.makedirs(hand_surface_pcd_save_dir_path,exist_ok=True)
    os.makedirs(hand_mesh_save_dir_path,exist_ok=True)
    os.makedirs(orginal_hand_mesh_save_dir_path,exist_ok=True)
    data_save_index = 0
    for index, data in tqdm(enumerate(dataloader), total= len(dataloader)):
        for data_index in torch.arange(data["obj_pcd"].shape[0]):
            # if data["sub_name"][data_index] == "1" and data["obj_name"][data_index] == "oreo_medium":
            hand_surface_points = data["hand_surface_points"][data_index].to(device)
            obj_pcd = data["obj_pcd"][data_index].to(device)
            obj_pcd_normal = data["obj_pcd_normal"][data_index].to(device)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(data["obj_pcd"][data_index].cpu().numpy().reshape((-1,3)))
            pcd.normals = o3d.utility.Vector3dVector(data["obj_pcd_normal"][data_index].cpu().numpy().reshape((-1,3)))
            o3d.io.write_point_cloud(pjoin(obj_pcd_save_dir_path,f"pcd_{data_save_index}.ply"),pcd)

            one_qpos_trans = DynamicGrasp.trans_denormalize(data["hand_qpos"][data_index][:3])
            one_qpos_joint_angle = DynamicGrasp.angle_denormalize(data["hand_qpos"][data_index][3:])

            original_hand_mesh = hand_model.get_meshes_from_q((torch.cat([one_qpos_trans,one_qpos_joint_angle],dim = 0)).unsqueeze(0).to(device))
            original_hand_mesh.export(pjoin(hand_mesh_save_dir_path,f"original_hand_{data_save_index}.ply"))


            original_orginal_hand_mesh = hand_model.get_meshes_from_q(data["original_hand_qpos"][data_index].unsqueeze(0).to(device))
            original_orginal_hand_mesh.export(pjoin(orginal_hand_mesh_save_dir_path,f"original_hand_{data_save_index}.ply"))

            hand_surface_pcd = o3d.geometry.PointCloud()
            hand_surface_pcd.points = o3d.utility.Vector3dVector(data["hand_surface_points"][data_index].cpu().numpy().reshape((-1,3)))
            o3d.io.write_point_cloud(pjoin(hand_surface_pcd_save_dir_path,f"hand_surface_pcd_{data_save_index}.ply") ,hand_surface_pcd)

            data_save_index += 1

    # for index, data in tqdm(enumerate(dataloader), total= len(dataloader)):

    #         hand_surface_points = data["hand_surface_points"].to(device)
    #         obj_pcd = data["obj_pcd"].to(device)
    #         obj_pcd_normal = data["obj_pcd_normal"].to(device)
    #         # pen_value, max_col_index = pen_loss(obj_pcd,obj_pcd_normal,hand_surface_points)
    #         obj_pcd_distance_diff, obj2hand_dist, idx1, idx2 = chd(hand_surface_points,obj_pcd)# [1,pcd_num]
    #         # print(obj2hand_dist.shape)

    #         every_pair_min_dist = torch.min(obj2hand_dist,dim = 1).values
    #         max_dis_index = torch.argmax(every_pair_min_dist)
    #         print(index)
    #         print(every_pair_min_dist[max_dis_index])
    #         print(data["sub_name"][max_dis_index])
    #         print(data["obj_name"][max_dis_index])
    #         # print(pen_value, max_col_index)

    #         pcd = o3d.geometry.PointCloud()
    #         pcd.points = o3d.utility.Vector3dVector(data["obj_pcd"][max_dis_index].cpu().numpy().reshape((-1,3)))
    #         pcd.normals = o3d.utility.Vector3dVector(data["obj_pcd_normal"][max_dis_index].cpu().numpy().reshape((-1,3)))
    #         o3d.io.write_point_cloud(pjoin(obj_pcd_save_dir_path,f"pcd_{data_save_index}.ply"),pcd)

    #         one_qpos_trans = DynamicGrasp.trans_denormalize(data["hand_qpos"][max_dis_index][:3])
    #         one_qpos_joint_angle = DynamicGrasp.angle_denormalize(data["hand_qpos"][max_dis_index][3:])

    #         original_hand_mesh = hand_model.get_meshes_from_q((torch.cat([one_qpos_trans,one_qpos_joint_angle],dim = 0)).unsqueeze(0).to(device))
    #         original_hand_mesh.export(pjoin(hand_mesh_save_dir_path,f"original_hand_{data_save_index}.ply"))

    #         hand_surface_pcd = o3d.geometry.PointCloud()
    #         hand_surface_pcd.points = o3d.utility.Vector3dVector(data["hand_surface_points"][max_dis_index].cpu().numpy().reshape((-1,3)))
    #         o3d.io.write_point_cloud(pjoin(hand_surface_pcd_save_dir_path,f"hand_surface_pcd_{data_save_index}.ply") ,hand_surface_pcd)

    #         data_save_index += 1
        # print(dis_loss(hand_surface_points,obj_pcd))

    # save_dir_path = "./test_meshes/DynamicGrasp_dataset"
    # os.makedirs(save_dir_path, exist_ok= True)


    # obj_pcd_save_dir_path = pjoin(save_dir_path,f"obj_pcd")
    # hand_surface_pcd_save_dir_path = pjoin(save_dir_path,f"hand_surface_pcd")
    # hand_mesh_save_dir_path = pjoin(save_dir_path,f"hand_mesh")

    # os.makedirs(obj_pcd_save_dir_path,exist_ok=True)
    # os.makedirs(hand_surface_pcd_save_dir_path,exist_ok=True)
    # os.makedirs(hand_mesh_save_dir_path,exist_ok=True)


    # for index, data in tqdm(enumerate(dataloader),desc="saving data"):

    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(data["obj_pcd"].cpu().numpy().reshape((-1,3)))
    #     pcd.normals = o3d.utility.Vector3dVector(data["obj_pcd_normal"].cpu().numpy().reshape((-1,3)))
    #     o3d.io.write_point_cloud(pjoin(obj_pcd_save_dir_path,f"pcd_{index}.ply"),pcd)

    #     original_hand_mesh = hand_model.get_meshes_from_q(data["hand_qpos"].to(device))
    #     original_hand_mesh.export(pjoin(hand_mesh_save_dir_path,f"original_hand_{index}.ply"))

    #     hand_surface_pcd = o3d.geometry.PointCloud()
    #     hand_surface_pcd.points = o3d.utility.Vector3dVector(data["hand_surface_points"].cpu().numpy().reshape((-1,3)))
    #     o3d.io.write_point_cloud(pjoin(hand_surface_pcd_save_dir_path,f"hand_surface_pcd_{index}.ply") ,hand_surface_pcd)

