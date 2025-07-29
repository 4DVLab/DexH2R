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
from utils.rot6d import  uniform_rotation_matrics
from os.path import join as pjoin
from pytorch3d.transforms import matrix_to_axis_angle
import hydra
from omegaconf import DictConfig
from models.base import create_model
from tqdm import tqdm
import trimesh as tm


# export LD_LIBRARY_PATH=~/miniconda3/envs/dynamic_grasp_release/lib/:$LD_LIBRARY_PATH

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



@hydra.main(version_base=None, config_path="./../configs", config_name="default")
def main(cfg: DictConfig) -> None:

    #load model 
    device = "cuda"
    model = create_model(cfg, device=device)
    model.to(device=device)
    load_ckpt(model, path=os.path.join(cfg.pretrain_model_dir_path, f'model_{cfg.pretrain_model_index}.pth'))   
    model.eval()
    device = model.device


    data_dir_path = f"./../dataset/obj_surface_pcd_{cfg.task.dataset.original_points_num}/"
    all_model_name_list = [model_name[:-4] for model_name in os.listdir(data_dir_path)]

    process_data_batch_size = 60
    self_rotate_pcd_num = 20

    global_rotate_num = 25
    
    ksample = global_rotate_num *  self_rotate_pcd_num  
    with torch.no_grad():
        with tqdm(enumerate(all_model_name_list),desc = "final qpose data gen") as pbar:
            for model_name in tqdm(all_model_name_list):
                pbar.set_description(model_name)                
                data_qpose = []

                pcd_surface_data_path = pjoin(data_dir_path,f"{model_name}.ply")
                pcd_surface_data = torch.from_numpy(np.array(tm.load(pcd_surface_data_path).vertices)).view(-1,3).to(torch.float32).to(device).unsqueeze(0)#[1,pcd_n,3]
                rotation_tensors = uniform_rotation_matrics(global_rotate_num,self_rotate_pcd = pcd_surface_data[0],self_rotate_pcd_num = self_rotate_pcd_num).to(torch.float32).to(device)
                
                for data_index_begin in torch.arange(0,pcd_surface_data.shape[0],process_data_batch_size):

                    batch_pcd_data = pcd_surface_data[data_index_begin:data_index_begin + process_data_batch_size].to(device)# [seq_len,num_pcd,3]

                    mean_point = batch_pcd_data.mean(dim = 1, keepdim = True)# [data_size,1,3]
                    batch_pcd_data = batch_pcd_data - mean_point
                    data_size = batch_pcd_data.shape[0]# maybe the data is not enough the real batch size 

                    obj_pcd_rot = torch.matmul(batch_pcd_data.unsqueeze(-2).unsqueeze(1), rotation_tensors.unsqueeze(1).unsqueeze(0)).reshape(data_size * ksample,-1,3)

                    data = {'obj_pcd': obj_pcd_rot}

                    model_predict_qpose,_ = model.sample(data, k_sample=data_size * ksample)#.to(torch.float32).view(data_size,ksample,27)# [batch_size,ksample,27]
                    
                    model_predict_qpose = model_predict_qpose.to(torch.float32).view(data_size,ksample,27)
                    qpose_trans = (model_predict_qpose[...,:3].unsqueeze(-2) @ rotation_tensors.permute(0,2,1).unsqueeze(0))# [data_size,ksample,1,3], [1,ksample,3,3] -> [data_size,ksample,1,3]
                    qpose_trans = (qpose_trans +  mean_point.unsqueeze(1)).squeeze(-2)# add ksample size
                    qpose_axis_angle = matrix_to_axis_angle(rotation_tensors).unsqueeze(0).repeat((data_size,1,1))# [data_size,ksample,3]
                    
                    one_data_qpose = torch.cat([qpose_trans,qpose_axis_angle,model_predict_qpose[...,3:]],dim = -1)
                    data_qpose.append(one_data_qpose.cpu())

                data_qpose = torch.cat(data_qpose,dim = 0).squeeze(0)
            
                save_dir_path = "./../dataset/model_final_qpose_cvae/"
                os.makedirs(save_dir_path,exist_ok=True)
                save_path = pjoin(save_dir_path,f"{model_name}_final_qpose.pt")
                torch.save(data_qpose,save_path) 
                








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

    