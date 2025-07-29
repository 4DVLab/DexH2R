import os
import json
import numpy as np
import cv2
import torch
import open3d as o3d  # 用于处理点云和保存为 .ply 格式
import time
import yaml
from natsort import natsorted 
import argparse
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix,axis_angle_to_matrix, matrix_to_euler_angles


# create argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--grasping_period', action='store_true',  # 如果指定该参数则设为 True
                   help="Whether to include grasping period (default: False)")
# parse arguments
args = parser.parse_args()
# use arguments
grasping_period = args.grasping_period

merge_dir = "../dataset/DexH2R_merge_dataset"
# check if the folder exists
if not os.path.exists(merge_dir):
    os.makedirs(merge_dir)
    print(f"folder {merge_dir} has been created.")
else:
    print(f"folder {merge_dir} already exists.")

# load JSON config file
data_split_path = "../dataset/dataset_split.json"
with open(data_split_path, 'r') as f:
    config = json.load(f)

# define the root path of the dataset
dataset_root = '../dataset/DexH2R_dataset'
objpcd_dir = "../dataset/obj_surface_pcd_4096"

# get the sub-names of the dataset
train_sub_names = config['train_sub_names']
val_sub_names = config['val_sub_names']
test_sub_names = config['test_sub_names']

# get the object names of the dataset
train_obj = config['train_obj']
val_obj = config['val_obj']
test_obj = config['test_obj']


# TODO: 修改dataset的划分
val_obj_1 = val_obj
val_obj_2 = val_obj
val_obj_3 = train_obj
val_name_1 = val_sub_names
val_name_2 = train_sub_names
val_name_3 = val_sub_names

test_obj_1 = train_obj + val_obj
test_obj_2 = test_obj
test_obj_3 = test_obj
test_name_1 = test_sub_names
test_name_2 = test_sub_names
test_name_3 = train_sub_names + val_sub_names

# # rgb and depth image resize
# size_1, size_2 = 180, 320

def load_grasp_begin_index(grasp_data_path:str):
    yaml_file_path = os.path.join(grasp_data_path,"motion_info.yaml")
    with open(yaml_file_path,"r") as f:
        grasp_begin_index = int(yaml.load(f,Loader=yaml.FullLoader)["grasp_begin_frame_index"])
    return grasp_begin_index


def find_nearest_final_qpose(qpose,target_qpose_pool):
    '''
    param:
        qpose:              [seq_len,30]
        target_qpose_pool   [seq_len,30,30]
    return 
        selected_final_qpose [seq_len,30]
    '''
    wind_len = target_qpose_pool.shape[1]
    seq_len = qpose.shape[0]
    # use euler angle to find the nearest final qpose
    qpose_euler = matrix_to_euler_angles(axis_angle_to_matrix(qpose[...,3:6].float()),'XYZ')
    target_qpose_pool_euler = matrix_to_euler_angles(axis_angle_to_matrix(target_qpose_pool[...,3:6].float()),'XYZ')
    qpose_euler_diff = (qpose_euler.unsqueeze(1)[...,:3].float() - target_qpose_pool_euler[...,:3].float()).abs().sum(-1)# [seq_len,30]
    # only use trans
    trans_diff = (qpose.unsqueeze(1)[...,:3].float() - target_qpose_pool[...,:3].float()).abs().sum(-1)# [seq_len,30]
    top_k = 40
    _, top_k_indices = torch.topk(trans_diff, top_k, dim=1, largest=False)  # [seq_len,5]
    batch_indices = torch.arange(seq_len).unsqueeze(1).expand(-1, top_k)  # [seq_len,5]
    filtered_trans_diff = qpose_euler_diff[batch_indices, top_k_indices]  # [seq_len,5]
    _, min_local_indices = torch.min(filtered_trans_diff, dim=1)  # [seq_len]
    # get the final index (from top_k_indices)
    final_indices = top_k_indices[torch.arange(seq_len), min_local_indices]  # [seq_len]
    selected_final_qpose = target_qpose_pool[torch.arange(seq_len), final_indices, :]
    return selected_final_qpose,final_indices



# 定义一个函数，用于加载特定子集和物品的数据
def load_data(sub_names, obj_names, episode_start, number, grasping_period):
    data = []
    qpose_data = []
    qpose_delta_data = []
    final_grasp_data = []
    final_grasp_group_data = []
    objpcd_intact_4096_data = []
    objpcd_normal_intact_4096_data = []
    obs_objpcd_4096_data = []
    episode_data = []
    traj_index_data = []
    
    episode_count = episode_start
    number = number
    for sub_name in sub_names:
        sub_name = str(sub_name)
        sub_dir = os.path.join(dataset_root, sub_name)
        if not os.path.exists(sub_dir):
            print(f"Warning: {sub_name} does not exist in dataset.")
            continue
        for obj_name in obj_names:
            obj_dir = os.path.join(sub_dir, obj_name)
            if os.path.exists(obj_dir):
                # traverse all the trajectories in obj_dir
                sorted_folders = natsorted(os.listdir(obj_dir))
                for folder_name in sorted_folders:
                    traj_dir = os.path.join(obj_dir, folder_name)
                    if os.path.exists(traj_dir):
                        grasp_begin_index = load_grasp_begin_index(traj_dir)
                        qpose_path = os.path.join(traj_dir, "qpos.pt")
                        final_grasp_group_path = os.path.join(traj_dir, "model_final_qpose_cvae.pt")
                        obj_pose_path = os.path.join(traj_dir, "obj_pose.pt")
                        obj_downsample_ply_path = f"{objpcd_dir}/{obj_name}.ply"
                        pcd_path = os.path.join(traj_dir, "real_obj_pcd_xyz.pt")
                        qpose_array = torch.load(qpose_path).cpu().numpy()
                        # import pdb; pdb.set_trace()
                        if not grasping_period:
                            traj_end = grasp_begin_index
                        else:
                            traj_end = qpose_array.shape[0]
                        qpose_array = qpose_array[:traj_end]
                        final_qpose_group = torch.load(final_grasp_group_path).cpu().numpy()[:traj_end]
                        final_pose_num = final_qpose_group.shape[1]
                        pcd_array = torch.load(pcd_path).cpu().numpy()[:traj_end]
                        obj_pose_array = torch.load(obj_pose_path).cpu().numpy()[:traj_end]
                        objpcd_intact_cloud = o3d.io.read_point_cloud(obj_downsample_ply_path)
                        
                        episode_count += traj_end
                        number += 1
                        episode_data.append(episode_count)
                        
                        for i in range(traj_end):
                            tmp_cloud = o3d.geometry.PointCloud()
                            tmp_cloud.points = o3d.utility.Vector3dVector(np.asarray(objpcd_intact_cloud.points))
                            tmp_cloud.normals = o3d.utility.Vector3dVector(np.asarray(objpcd_intact_cloud.normals))
                            obj_pose = obj_pose_array[i]
                            objpcd_intact_cloud_transformed = tmp_cloud.transform(obj_pose)
                            objpcd_intact_vertices_transformed = np.asarray(objpcd_intact_cloud_transformed.points)
                            objpcd_intact_normals_transformed = np.asarray(objpcd_intact_cloud_transformed.normals)
                            obs_objpcd = pcd_array[i]
                            objpcd_intact_4096_data.append(objpcd_intact_vertices_transformed)
                            objpcd_normal_intact_4096_data.append(objpcd_intact_normals_transformed)
                            obs_objpcd_4096_data.append(obs_objpcd)
                            if i == 0:
                                qpose_delta_data.append(np.concatenate((qpose_array[0][:6]-qpose_array[0][:6], qpose_array[0][8:]-qpose_array[0][8:])))
                            else:
                                qpose_delta_data.append(np.concatenate((qpose_array[i][:6]-qpose_array[i-1][:6], qpose_array[i][8:]-qpose_array[i-1][8:])))
                            qpose_data.append(np.concatenate((qpose_array[i][:6], qpose_array[i][8:])))
                            traj_index_data.append(number)
                            chosen_final_grasp, _ = find_nearest_final_qpose(torch.tensor(qpose_array[i]).unsqueeze(0), torch.tensor(final_qpose_group[i]).unsqueeze(0))
                            chosen_final_grasp = chosen_final_grasp.squeeze(0).cpu().numpy()
                            final_grasp_data.append(np.concatenate((chosen_final_grasp[:6], chosen_final_grasp[8:])))
                            final_grasp_group_data.append(np.concatenate((final_qpose_group[i][:,:6], final_qpose_group[i][:,8:]), axis=1))
                        
                        print("obj_dir = ", obj_dir)
                        print("episode_count = ", episode_count)
            else:
                print(f"Warning: {obj_name} not found for {sub_name}.")
    data.append(qpose_data)
    data.append(qpose_delta_data)
    data.append(final_grasp_data)
    data.append(final_grasp_group_data)
    data.append(objpcd_intact_4096_data)
    data.append(objpcd_normal_intact_4096_data)
    data.append(obs_objpcd_4096_data)
    data.append(episode_data)
    data.append(traj_index_data)
    return data, episode_count, number

train_data, train_episode_end, train_num = load_data(train_sub_names, train_obj, 0, 0, grasping_period)
val_data_1, val_episode_end_1, val_num_1 = load_data(val_name_1, val_obj_1, train_episode_end, 0, grasping_period)
val_data_2, val_episode_end_2, val_num_2 = load_data(val_name_2, val_obj_2, val_episode_end_1, val_num_1, grasping_period)
val_data_3, val_episode_end_3, val_num_3 = load_data(val_name_3, val_obj_3, val_episode_end_2, val_num_2, grasping_period)
test_data_1, test_episode_end_1, test_num_1 = load_data(test_name_1, test_obj_1, val_episode_end_3, 0, grasping_period)
test_data_2, test_episode_end_2, test_num_2 = load_data(test_name_2, test_obj_2, test_episode_end_1, test_num_1, grasping_period)
test_data_3, test_episode_end_3, test_num_3 = load_data(test_name_3, test_obj_3, test_episode_end_2, test_num_2, grasping_period)

print("train_num = ", train_num)
print("val_num_1 = ", val_num_1)
print("val_num_2 = ", val_num_2)
print("val_num_3 = ", val_num_3)
print("test_num_1 = ", test_num_1)
print("test_num_2 = ", test_num_2)
print("test_num_3 = ", test_num_3)
print("val_num = ", val_num_1+val_num_2+val_num_3)
print("test_num = ", test_num_1+test_num_2+test_num_3)

print("train_episode_end = ", train_episode_end)
print("val_episode_end_1 = ", val_episode_end_1)
print("val_episode_end_2 = ", val_episode_end_2)
print("val_episode_end_3 = ", val_episode_end_3)
print("test_episode_end_1 = ", test_episode_end_1)
print("test_episode_end_2 = ", test_episode_end_2)
print("test_episode_end_3 = ", test_episode_end_3)

val_data = val_data_1
test_data = test_data_1
for i in range(len(train_data)):
    val_data[i] += val_data_2[i]
    val_data[i] += val_data_3[i]
    test_data[i] += test_data_2[i]
    test_data[i] += test_data_3[i]

# TODO: can modify the code to only generate the training or validation or test dataset
merge_qpose_array = np.array(train_data[0] + val_data[0] + test_data[0])
merge_action_array = np.array(train_data[0] + val_data[0] + test_data[0])
merge_qpose_delta_array = np.array(train_data[1] + val_data[1] + test_data[1])
merge_final_grasp_array = np.array(train_data[2] + val_data[2] + test_data[2])
merge_final_grasp_group_array = np.array(train_data[3] + val_data[3] + test_data[3])
merge_objpcd_intact_4096_array = np.array(train_data[4] + val_data[4] + test_data[4])
merge_objpcd_normal_intact_4096_array = np.array(train_data[5] + val_data[5] + test_data[5])
merge_obs_objpcd_4096_array = np.array(train_data[6] + val_data[6] + test_data[6])
episode_array = np.array(train_data[7] + val_data[7] + test_data[7])
merge_traj_index_array = np.array(train_data[8] + val_data[8] + test_data[8])


print("merge_qpose_array.shape = ", merge_qpose_array.shape)
print("merge_action_array.shape = ", merge_action_array.shape)
print("merge_qpose_delta_array.shape = ", merge_qpose_delta_array.shape)
print("merge_final_grasp_array.shape = ", merge_final_grasp_array.shape)
print("merge_final_grasp_group_array.shape = ", merge_final_grasp_group_array.shape)
print("merge_objpcd_intact_4096_array.shape = ", merge_objpcd_intact_4096_array.shape)
print("merge_objpcd_normal_intact_4096_array.shape = ", merge_objpcd_normal_intact_4096_array.shape)
print("merge_obs_objpcd_4096_array.shape = ", merge_obs_objpcd_4096_array.shape)
print("episode_array.shape = ", episode_array.shape)
print("merge_traj_index_array.shape = ", merge_traj_index_array.shape)


np.save(f"{merge_dir}/agent_pos.npy", merge_qpose_array)
np.save(f"{merge_dir}/velocity.npy", merge_qpose_delta_array)
np.save(f"{merge_dir}/final_grasp.npy", merge_final_grasp_array)
np.save(f"{merge_dir}/final_grasp_group.npy", merge_final_grasp_group_array)
np.save(f"{merge_dir}/objpcd_intact.npy", merge_objpcd_intact_4096_array)
np.save(f"{merge_dir}/objpcd_normal_intact.npy", merge_objpcd_normal_intact_4096_array)
np.save(f"{merge_dir}/obs_objpcd.npy", merge_obs_objpcd_4096_array)
np.save(f"{merge_dir}/episode_ends.npy", episode_array)
np.save(f"{merge_dir}/traj_index.npy", merge_traj_index_array)