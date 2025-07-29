import os
import torch
import sys
sys.path.append(os.getcwd())
import numpy as np
from os.path import join as pjoin
from pytorch3d.transforms import matrix_to_axis_angle
from utils.utils import safe_search_file
from tqdm import tqdm
from pytorch3d.transforms import axis_angle_to_matrix
import argparse
from datasets.DynamicMotion import load_seq_data_path
from concurrent.futures import ThreadPoolExecutor

def arg_init():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--final_qpos_folder_name", default="model_final_qpose_cvae", help="model_final_qpose_cvae"
    )

    args = parser.parse_args()
    return args

def save_one_seq_final_pose(data_dir_path, obj_final_qpose_data_dict, final_qpos_folder_name):
    obj_pose_path = pjoin(data_dir_path,"obj_pose.pt")# [seq_len,4,4]
    obj_pose = torch.load(obj_pose_path).to(torch.float32)

    hand_transform = (obj_final_qpose_data_dict[0]).unsqueeze(0).repeat(obj_pose.shape[0],1,1,1)
    hand_joint_angle = (obj_final_qpose_data_dict[1]).unsqueeze(0).repeat(obj_pose.shape[0],1,1) #[seq_len,final_qpose_num,joint_angle_dim]
    
    obj_pose = obj_pose.unsqueeze(1).repeat(1,hand_transform.shape[1],1,1)

    seq_final_qpose_transform = obj_pose @ hand_transform
    seq_final_qpose_translation = seq_final_qpose_transform[...,:3,3]
    seq_final_qpose_rotation = matrix_to_axis_angle(seq_final_qpose_transform[...,:3,:3])
    
    seq_final_qpose = torch.cat([seq_final_qpose_translation,seq_final_qpose_rotation,hand_joint_angle],dim=-1).to(torch.float16)
    
    seq_final_qpose_path = pjoin(data_dir_path,f"{final_qpos_folder_name}.pt")
    torch.save(seq_final_qpose,seq_final_qpose_path)


def main():

    args = arg_init()
    final_qpos_folder_name = args.final_qpos_folder_name

    motion_data_dir_path = "./../dataset/DexH2R_dataset/"
    all_grasp_data_dir_path = load_seq_data_path(motion_data_dir_path)


    all_grasp_data_parent_dir_path = [str(item_path.parent) for item_path in all_grasp_data_dir_path]

    path_obj_name_list = [item_path.split("/")[-2] for item_path in all_grasp_data_parent_dir_path]

    model_final_qpose_data_dir_path = f"./../dataset/{final_qpos_folder_name}/"

    model_name_list = [file_name.replace("_final_qpose.pt","") for file_name in os.listdir(model_final_qpose_data_dir_path)]
    final_qpose_data_dict = {}

    for model_name in model_name_list:
        final_qpose_data_path = pjoin(model_final_qpose_data_dir_path,f"{model_name}_final_qpose.pt")   
        final_qpose_data = torch.load(final_qpose_data_path)
        transform_matrix = torch.eye(4).unsqueeze(0).repeat(final_qpose_data.shape[0],1,1)# [B,4,4] 500
        transform_matrix[:,:3,:3] = axis_angle_to_matrix(final_qpose_data[:,3:6])
        transform_matrix[:,:3,3] = final_qpose_data[:,:3]
        final_qpose_data_dict[model_name] = [transform_matrix,final_qpose_data[:,6:]]


    func_arg_list = [] 
    for idx,data_dir_path in tqdm(enumerate(all_grasp_data_parent_dir_path),total= len(all_grasp_data_parent_dir_path)):
        obj_name = path_obj_name_list[idx]
        obj_final_qpose_data_dict = final_qpose_data_dict[obj_name]
        func_arg_list.append((data_dir_path, obj_final_qpose_data_dict, final_qpos_folder_name))
    
    with ThreadPoolExecutor(max_workers=12) as executor:  # 可以根据系统调整worker数量
        list(tqdm(executor.map(lambda x: save_one_seq_final_pose(*x), func_arg_list), total=len(func_arg_list), desc="save seq final grasp pose"))


        
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
