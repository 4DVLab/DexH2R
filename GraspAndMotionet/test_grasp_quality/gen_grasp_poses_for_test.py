# import debugpy
# debugpy.listen(("localhost", 15000))
# debugpy.wait_for_client()  

import os
import sys
# os.chdir("./../")
sys.path.append(os.getcwd())

import torch
import numpy as np
from omegaconf import DictConfig
import random
from os.path import join as pjoin
from pytorch3d.transforms import matrix_to_axis_angle
import hydra
from omegaconf import DictConfig
from models.base import create_model
from tqdm import tqdm
import trimesh as tm

from utils.rot6d import compute_rotation_matrix_from_ortho6d,rot_to_orthod6d
from utils.handmodel import get_handmodel
from collections import defaultdict

def random_rot(device='cuda'):
    rot_angles = np.random.random(3) * np.pi * 2
    theta_x, theta_y, theta_z = rot_angles[0], rot_angles[1], rot_angles[2]
    Rx = torch.tensor([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]]).to(device)
    Ry = torch.tensor([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]]).to(device)
    Rz = torch.tensor([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]]).to(device)
    return (Rx @ Ry @ Rz).clone().detach()  # [3, 3]



def load_ckpt(model: torch.nn.Module, path: str) -> None:
    """ load ckpt for current model

    Args:
        model: current model
        path: save path
    """
    assert os.path.exists(path), 'Can\'t find provided ckpt.'
    saved_state_dict = torch.load(path)['model']
    model_state_dict = model.state_dict()
    for key in model_state_dict:
        if key in saved_state_dict:
            model_state_dict[key] = saved_state_dict[key]
        ## model is trained with ddm
        if 'module.'+key in saved_state_dict:
            model_state_dict[key] = saved_state_dict['module.'+key]
    
    model.load_state_dict(model_state_dict)


@hydra.main(version_base=None, config_path="./../configs", config_name="grasp")
def main(cfg: DictConfig) -> None:

    #load model 
    device = "cuda"
    model = create_model(cfg, device=device)
    model.to(device=device)
    load_ckpt(model, path=os.path.join(cfg.pretrain_model_dir_path, f'model_{cfg.pretrain_model_index}.pth'))   
    model.eval()


    data_dir_path = f"./../dataset/obj_surface_pcd_4096/"
    all_model_name_list = [model_name[:-4] for model_name in os.listdir(data_dir_path)]

    ksample = 25
    e3m5_orginal_trans = torch.tensor([ 0.0000, -0.0100,  0.2470]).to(device)
    res_data = defaultdict(dict)
    
    # hand_model = get_handmodel(batch_size=1, device=device)
    # output_grasp_mesh_dir_path = "test_meshes/viz_grasp"
    # os.makedirs(output_grasp_mesh_dir_path,exist_ok=True)
    # data_index = 0

    with torch.no_grad():

        with tqdm(enumerate(all_model_name_list),desc = "final qpose data gen") as pbar:
            for model_name in tqdm(all_model_name_list):
                pbar.set_description(model_name)                


                rot_mat = torch.stack([random_rot(device=device) for _ in torch.arange(ksample)]).to(torch.float32)# [ksample,3,3]

                pcd_surface_data_path = pjoin(data_dir_path,f"{model_name}.ply")
                pcd_surface_data = torch.from_numpy(np.array(tm.load(pcd_surface_data_path).vertices)).to(torch.float32).to(device).view(-1,3)#[1,pcd_n,3]

                obj_pcd_rot = (pcd_surface_data.unsqueeze(-2).unsqueeze(0) @ rot_mat.unsqueeze(1)).reshape(ksample,-1,3)

                data = {'obj_pcd': obj_pcd_rot}

                model_predict_qpose,_ = model.sample(data, k_sample=ksample)
                
                model_predict_qpose = model_predict_qpose.to(torch.float32).view(ksample,27)

                model_predict_qpose[...,:3] = model_predict_qpose[...,:3] - e3m5_orginal_trans # tranform the frame from the palm to the forearm

                qpose_trans = (model_predict_qpose[...,:3].unsqueeze(-2) @ rot_mat.permute(0,2,1)).squeeze(-2)# [ksample,1,3], [ksample,3,3] -> [ksample,3]
                hand_orthod6d = rot_to_orthod6d(rot_mat)# [ksample,6]
                
                output_qpose = torch.cat([qpose_trans,hand_orthod6d,model_predict_qpose[...,3:]],dim = -1)

                res_data['sample_qpos'][model_name] = output_qpose.cpu().numpy()
                
                # grasps['sample_qpos'][object_name]

                # print(model_name)
                # for qpos in (output_qpose):
                #     hand_mesh_list = hand_model.get_meshes_from_q(qpos.unsqueeze(0).to(device))
                #     hand_mesh = tm.util.concatenate(hand_mesh_list)
                #     hand_mesh.export(pjoin(output_grasp_mesh_dir_path,f"{data_index}.ply"))
                #     data_index += 1

                # exit(0)
    os.makedirs(cfg.grasp_gen_res_save_dir_path,exist_ok=True)
    torch.save(res_data,pjoin(cfg.grasp_gen_res_save_dir_path,"res_data.pt"))


if __name__ == '__main__':
    ## set random seed
    seed = 0
    torch.backends.cudnn.benchmark = False     
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    main()

    