from typing import Any, Tuple, Dict
import os
import sys
sys.path.append(os.getcwd())
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig, OmegaConf
import transforms3d
from datasets.misc import collate_fn_squeeze_pcd_batch_grasp, collate_fn_general
from datasets.transforms import make_default_transform
from datasets.base import DATASET
import json
from utils.registry import Registry
import trimesh as tm
from utils.e3m5_hand_model import get_e3m5_handmodel, pen_loss, dis_loss
from os.path import join as pjoin
from tqdm import tqdm
import open3d as o3d
from datasets.DynamicGrasp import DynamicGrasp


def load_from_json(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    return data["_train_split"], data["_test_split"], data["_all_split"]


@DATASET.register()
class DexGraspNet(Dataset):
    """ Dataset for pose generation, training with DexGraspNet Dataset
    """

    # read json
    input_file = "./../dataset/DexGraspNet/grasp.json"
    _train_split, _test_split, _all_split = load_from_json(input_file)
    # _train_split = random.sample(_train_split, 100)
    _joint_angle_lower = torch.tensor([-0.5235988, -0.7853982, -0.43633232, 0., 0., 0., -0.43633232, 0., 0., 0.,
                                       -0.43633232, 0., 0., 0., 0., -0.43633232, 0., 0., 0., -1.047, 0., -0.2618,
                                       -0.5237, 0.])
    _joint_angle_upper = torch.tensor([0.17453292, 0.61086524, 0.43633232, 1.5707964, 1.5707964, 1.5707964, 0.43633232,
                                       1.5707964, 1.5707964, 1.5707964, 0.43633232, 1.5707964, 1.5707964, 1.5707964,
                                       0.6981317, 0.43633232, 1.5707964,  1.5707964, 1.5707964, 1.047, 1.309, 0.2618,
                                       0.5237, 1.])

    _global_trans_lower = torch.tensor([-0.13128923, -0.10665303, -0.45753425])
    _global_trans_upper = torch.tensor([0.12772022, 0.22954416, -0.21764427])

    _NORMALIZE_LOWER = -1.
    _NORMALIZE_UPPER = 1.
    e3m5_orginal_trans = torch.tensor([ 0.0000, -0.0100,  0.2470])
    def __init__(self, cfg: DictConfig, phase: str, **kwargs: Dict) -> None:
        super(DexGraspNet, self).__init__()
        self.phase = phase
        if self.phase == 'train':
            self.split = self._train_split
        elif self.phase == 'test':
            self.split = self._test_split
        elif self.phase == 'all':
            self.split = self._all_split
        else:
            raise Exception('Unsupported phase.')
        self.use_e3m5_hand_model = cfg.use_e3m5_hand_model
        if self.use_e3m5_hand_model:
            self.e3m5_orginal_trans = torch.tensor([ 0.0000, -0.0100,  0.2470])
            self._global_trans_lower += self.e3m5_orginal_trans
            self._global_trans_upper += self.e3m5_orginal_trans

        self.use_e3m5_hand_model = cfg.use_e3m5_hand_model
        self.device = cfg.device
        self.modeling_keys = cfg.modeling_keys
        self.sample_points_num = cfg.sample_points_num
        self.use_color = cfg.use_color
        self.use_normal = cfg.use_normal
        self.normalize_x = cfg.normalize_x
        self.normalize_x_trans = cfg.normalize_x_trans
        self.obj_dim = int(3 + 3 * self.use_color + 3 * self.use_normal)
        self.transform = make_default_transform(cfg, phase)
        ## resource folders

        self.data_dir = os.path.join('./../dataset/DexGraspNet')
        self.scene_path = os.path.join('./../dataset/DexGraspNet', 'object_pcds_nors.pkl')
        self._joint_angle_lower = self._joint_angle_lower.cpu()
        self._joint_angle_upper = self._joint_angle_upper.cpu()
        self._global_trans_lower = self._global_trans_lower.cpu()
        self._global_trans_upper = self._global_trans_upper.cpu()
        self.hand_model = get_e3m5_handmodel()
        ## load data
        self._pre_load_data()

    def _pre_load_data(self) -> None:
        """ Load dataset

        Args:
            case_only: only load single case for testing, if ture, the dataset will be smaller.
                        This is useful in after-training visual evaluation.
        
        """
        self.frames = []
        self.scene_pcds = {}
        grasp_dataset = torch.load(os.path.join(self.data_dir, '0.01filter_shadowhand_downsample.pt' ))#   dexgraspnet_shadowhand_downsample.pt
        self.scene_pcds = pickle.load(open(self.scene_path, 'rb'))
        self.dataset_info = grasp_dataset['info']

        # pre-process the dataset info
        for obj in grasp_dataset['info']['num_per_object'].keys():
            if obj not in self.split:
                self.dataset_info['num_per_object'][obj] = 0

        for data_idx,mdata in tqdm(enumerate(grasp_dataset['metadata']),desc="loading data",total= len(grasp_dataset['metadata'])):

            hand_rot_mat = mdata['rotations'].numpy()
            joint_angle = mdata['joint_positions'].clone().detach()
            global_trans = mdata['translations'].clone().detach()
            if self.use_e3m5_hand_model:
                global_trans += self.e3m5_orginal_trans# from zym hanc configuration to e3m5 hand configuration

            original_mdata_qpos = torch.cat([global_trans, joint_angle], dim=0)
            # hand_surface_points = self.hand_model.get_surface_points(original_mdata_qpos.unsqueeze(0))            

            if self.normalize_x:
                joint_angle = self.angle_normalize(joint_angle)
            if self.normalize_x_trans:
                global_trans = self.trans_normalize(global_trans)
                
            mdata_qpos = torch.cat([global_trans, joint_angle], dim=0)
            if mdata['object_name'] in self.split:
                self.frames.append({'robot_name': 'shadowhand',
                                    'object_name': mdata['object_name'],
                                    'object_rot_mat': hand_rot_mat.T,
                                    'qpos': mdata_qpos,
                                    'original_qpose':original_mdata_qpos,
                                    'scale': mdata['scale'],
                                    },
                                    )


    def trans_normalize(self, global_trans: torch.Tensor):
        global_trans_norm = torch.div((global_trans - self._global_trans_lower), (self._global_trans_upper - self._global_trans_lower))
        global_trans_norm = global_trans_norm * (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) - (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        return global_trans_norm

    def trans_denormalize(self, global_trans: torch.Tensor):
        global_trans_denorm = global_trans + (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        global_trans_denorm /= (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER)
        global_trans_denorm = global_trans_denorm * (self._global_trans_upper - self._global_trans_lower) + self._global_trans_lower
        return global_trans_denorm

    def angle_normalize(self, joint_angle: torch.Tensor):
        joint_angle_norm = torch.div((joint_angle - self._joint_angle_lower), (self._joint_angle_upper - self._joint_angle_lower))
        joint_angle_norm = joint_angle_norm * (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) - (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        return joint_angle_norm

    def angle_denormalize(self, joint_angle: torch.Tensor):
        joint_angle_denorm = joint_angle + (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        joint_angle_denorm /= (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER)
        joint_angle_denorm = joint_angle_denorm * (self._joint_angle_upper - self._joint_angle_lower) + self._joint_angle_lower
        return joint_angle_denorm

    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, index: Any) -> Tuple:

        frame = self.frames[index]
        if 'hand_surface_points' not in self.frames[index].keys():
            self.frames[index]['hand_surface_points'] = self.hand_model.get_surface_points(self.frames[index]['original_qpose'].unsqueeze(0)).squeeze(0)

        scale = frame['scale']
        scene_id = frame['object_name']
        scene_rot_mat = frame['object_rot_mat']
        scene_pc = self.scene_pcds[scene_id]
        nor = np.einsum('mn, kn->km', scene_rot_mat, scene_pc[:,3:6])
        scene_pc = np.einsum('mn, kn->km', scene_rot_mat, scene_pc[:,:3])
        scene_pc= scene_pc * scale

        ## randomly resample points
        if self.phase != 'train':
            np.random.seed(0) # resample point cloud with a fixed random seed
        resample_indices = np.random.permutation(len(scene_pc))[:self.sample_points_num]
        scene_pc = torch.from_numpy(scene_pc[resample_indices]).to(torch.float32)
        nor = nor[resample_indices]
        ## format point cloud xyz and feature
        xyz = scene_pc[:, 0:3]
        nor = torch.from_numpy(nor[:, 0:3])

        grasp_qpos = (
            frame['qpos']
        )

        data = {
            'hand_qpos': grasp_qpos,
            'obj_pcd': xyz,
            'obj_pcd_normal': nor,
            'hand_surface_points':frame['hand_surface_points'],
            'original_hand_qpos':frame['original_qpose']
        }

        if self.transform is not None:
            data = self.transform(data, modeling_keys=self.modeling_keys)

        return data

    def get_dataloader(self, **kwargs):
        return DataLoader(self, **kwargs)



if __name__ == '__main__':
    config_path = "./configs/task/grasp_gen_ur.yaml"
    cfg = OmegaConf.load(config_path)
    dataloader = DexGraspNet(cfg.dataset, 'train').get_dataloader(batch_size=100,
                                                                                  collate_fn=collate_fn_general,
                                                                                  num_workers=0,
                                                                                  pin_memory=True,
                                                                                  shuffle=True)
    device = torch.device('cuda:0')
    hand_model = get_e3m5_handmodel(device = device)


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