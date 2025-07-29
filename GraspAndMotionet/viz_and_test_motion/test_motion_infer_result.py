import os
import torch
import sys
sys.path.append(os.getcwd())
import numpy as np
from os.path import join as pjoin
from tqdm import tqdm
from utils.e3m5_hand_model import get_e3m5_handmodel
from os.path import join as pjoin
from tqdm import tqdm
from os.path import join as pjoin
import pickle
from pytorch3d.ops import knn_points
import open3d as o3d
import argparse

def jud_path_in_split(grasp_data_path:str,split):
    '''
    different with the grasp jud
    '''
    path_item = grasp_data_path.split("/")
    sub_name = path_item[-3]
    obj_name = path_item[-2]
    if sub_name in split["sub_names"] and obj_name in split["obj_names"]:
        return True
    return False

def read_path_file(file_path):
    path_list = []
    
    # 使用 with 语句打开文件，自动处理文件关闭
    with open(file_path, 'r') as f:
        # 逐行读取文件内容
        for line in f:
            # 去除每行末尾的换行符并添加到列表中
            path_list.append(line.strip())
    
    return path_list



def load_seq_obj_data(gtdata_dir_path,obj_data):
    device = "cuda"
    obj_pose_path = pjoin(gtdata_dir_path,"obj_pose.pt")
    obj_pose = torch.load(obj_pose_path).unsqueeze(1).to(device).to(torch.float32) #[seq_len,1,4,4]
    obj_name = gtdata_dir_path.split("/")[-2]
    obj_xyz_nor = obj_data[obj_name]
    obj_xyz = torch.cat([obj_xyz_nor[:,0:3],torch.ones(obj_xyz_nor.shape[0],1)],dim=1).unsqueeze(-1).unsqueeze(0).to(device) #[1,n,4,1]
    obj_nor = obj_xyz_nor[:,3:6].unsqueeze(0).unsqueeze(-1).to(device) #[1,n,3]

    obj_xyz = (obj_pose @ obj_xyz).squeeze(-1)[...,:3] #[seq_len,n,3]
    obj_nor = (obj_pose[:,:,:3,:3] @ obj_nor).squeeze(-1) #[seq_len,n,3]
    pcd_data = torch.cat([obj_xyz,obj_nor],dim=-1).to(torch.float32)


    return pcd_data
    


def cal_traj_length(model_predict_traj_tensor):
    frames_1 = model_predict_traj_tensor[:-1,:3]  # shape: [b-1,3]
    frames_2 = model_predict_traj_tensor[1:,:3]   # shape: [b-1,3]
    diff_norms = torch.norm(frames_1 - frames_2, p=2, dim=1)  # shape: [b-1]
    traj_length = diff_norms.sum()

    return traj_length



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
    distances = distances.sqrt()
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


def load_obj_data():
    obj_data_dir_path = "./../dataset/obj_surface_pcd_4096"
    file_name_list = [file_name for file_name in os.listdir(obj_data_dir_path) if file_name.endswith(".ply")]
    obj_data = {}
    for file_name in file_name_list:
        obj_name = file_name.replace(".ply", "")
        obj_pcd_o3d = o3d.io.read_point_cloud(os.path.join(obj_data_dir_path, file_name))
        obj_pcd_array = np.asarray(obj_pcd_o3d.points)
        obj_pcd_normal_array = np.asarray(obj_pcd_o3d.normals)
        obj_data[obj_name] = torch.from_numpy(np.concatenate([obj_pcd_array,obj_pcd_normal_array],axis=-1)).to(torch.float32)
    return obj_data




def load_correct_hand_pcd(model_predict_traj_tensor, hand_model):
    '''
    Remove the error accumulation problem that may be caused by autoregressive models
    '''
    hand_surface_pcd = hand_model.get_surface_points(model_predict_traj_tensor.to("cuda"))
    not_valid_data_index = (hand_surface_pcd > 5).any(dim=1).any(dim=1).nonzero()
    if len(not_valid_data_index) == 0:
        first_has_nan_index = hand_surface_pcd[0].shape[0]
    else:
        first_has_nan_index = not_valid_data_index[0]
    hand_surface_pcd = hand_surface_pcd[:first_has_nan_index]

    return hand_surface_pcd



def arg_init():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir_name", default = "test", help="the dir name of the data you want to evaluate, test/ablation."
    )
    args = parser.parse_args()
    return args


def main():
    arg = arg_init()

    eval_data_path = f"./viz_motion_output/{arg.dir_name}/vis_data.pth"
    data = torch.load(eval_data_path)

    obj_data = load_obj_data()

    hand_model = get_e3m5_handmodel(remove_wrist = True,device= "cuda",more_surface_points=True)
    
    device = "cuda"


    all_infer_frames_list = []
    all_total_pener_frame_list = []
    traj_accum_length_list = []
    success_flag_list = []
    safety_flag_list = []
    all_mean_max_pener_depth_list = []
    all_safety_flag_list = []
    all_success_flag_list = []



    for data_idx in tqdm(np.arange(len(data))):
        data_sample = data[data_idx]                  

        remove_begin = slice(5,None)

        pcd_data = load_seq_obj_data(data_sample["gtdata_dir_path"],obj_data)[remove_begin]

        seq_len = pcd_data.shape[0]
        model_predict_traj_tensor = data_sample["model_predict_traj_tensor"][remove_begin]

        seq_infer_frame = model_predict_traj_tensor.shape[0]
        all_infer_frames_list.append(seq_infer_frame)

        model_infer_mask = torch.tensor(data_sample["infer_frame_flag_list"])[remove_begin]
        model_predict_traj_tensor = model_predict_traj_tensor[model_infer_mask]

        hand_surface_pcd = load_correct_hand_pcd(model_predict_traj_tensor, hand_model)

        valid_range_hand_pcd = slice(None,hand_surface_pcd.shape[0])
        valid_model_predict_traj_tensor = model_predict_traj_tensor[valid_range_hand_pcd] # not nan
        valid_range_pcd_data = pcd_data[valid_range_hand_pcd]

        pen_max_frame_data = element_wise_pen_loss(valid_range_pcd_data,hand_surface_pcd) 
        

        seq_total_pener_frame = (pen_max_frame_data > 0).nonzero().shape[0]
        
        all_total_pener_frame_list.append(seq_total_pener_frame)
        
        sum_pen_frame_data = pen_max_frame_data.sum() 
        mean_max_pen_frame_data = sum_pen_frame_data / valid_model_predict_traj_tensor.shape[0] 
        all_mean_max_pener_depth_list.append(mean_max_pen_frame_data)
        
        seq_traj_accum_length = cal_traj_length(valid_model_predict_traj_tensor)
        traj_accum_length_list.append(seq_traj_accum_length)

        safety_flag = 0 if seq_total_pener_frame > 0 else 1
        success_flag = 1 if any([not x for x in data_sample["infer_frame_flag_list"]]) else 0 # if real traj len < seq_pcd len , so  success
        all_safety_flag_list.append(safety_flag)
        all_success_flag_list.append(success_flag)



        
    mean_infer_frames = np.mean(all_infer_frames_list)
    mean_total_pener_frame = np.mean(all_total_pener_frame_list)


    print("------------------------")
    print(f"final statics")
    print(f"mean infer frames {mean_infer_frames}")
    print(f"mean total pener frame {mean_total_pener_frame}")
    print(f"mean traj accum length {np.mean(traj_accum_length_list)}")
    print(f"mean mean max pener depth {torch.mean(torch.tensor(all_mean_max_pener_depth_list))}")
    print(f"mean safety flag {np.mean(all_safety_flag_list)}")
    print(f"new safety rate {(mean_infer_frames - mean_total_pener_frame) / mean_infer_frames}")
    print(f"mean success flag {np.mean(all_success_flag_list)}")
    print("------------------------")
        



if __name__ == "__main__":
    main()