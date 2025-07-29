from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
from os.path import join as pjoin
import cv2
from tqdm import tqdm
from time import time
import json
import yaml
from natsort import natsorted 
import argparse
import numpy as np
import torch

# create argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--grasping_period', action='store_true',  # 如果指定该参数则设为 True
                   help="Whether to include grasping period (default: False)")
args = parser.parse_args()
grasping_period = args.grasping_period

merge_dir = "../dataset/DexH2R_merge_dataset"
    
# define the target size of the resized image
size_1, size_2 = 180, 320

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
    
    

def sorted_dir_file_path(dir_path, file_type="jpg"):
    
    file_name_list = [file_name for file_name in os.listdir(dir_path) if file_name.endswith(file_type)]
    file_name_list.sort(key=lambda x: int(x.split(".")[0]))
    file_path_list = [pjoin(dir_path, file_name) for file_name in file_name_list]
    return file_path_list

def load_single_depth_img(img_path):
    new_size = (216, 384)
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img_resized = cv2.resize(img, new_size, interpolation=cv2.INTER_NEAREST)
    return img_resized

def load_single_rgb_img(img_path):
    new_size = (216, 384)
    img = cv2.imread(img_path)  
    img_resized = cv2.resize(img, new_size, interpolation=cv2.INTER_NEAREST)
    return img_resized

def load_grasp_begin_index(grasp_data_path:str):
    yaml_file_path = os.path.join(grasp_data_path,"motion_info.yaml")
    with open(yaml_file_path,"r") as f:
        grasp_begin_index = int(yaml.load(f,Loader=yaml.FullLoader)["grasp_begin_frame_index"])
    return grasp_begin_index



def load_one_folder_rgbd_path_list(kinect_dir_path, traj_end):
    '''
    param:
        kinect_1_dir_path: str
    '''
    if not os.path.exists(kinect_dir_path):
        return [], []
    
    kinect_rgb_dir_path = pjoin(str(kinect_dir_path), "rgb")
    kinect_depth_dir_path = pjoin(str(kinect_dir_path), "depth")

    kinect_rgb_path_list = sorted_dir_file_path(kinect_rgb_dir_path, file_type="jpg")
    kinect_depth_path_list = sorted_dir_file_path(kinect_depth_dir_path, file_type="png")
    return kinect_rgb_path_list[:traj_end], kinect_depth_path_list[:traj_end]   

# 加载RGB图像并进行处理
def process_rgb_image(img_path):
    img = cv2.imread(img_path)
    # 颜色空间转换：BGR转RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 调整尺寸
    resized_rgb = cv2.resize(rgb_img, (size_2, size_1))
    return resized_rgb

# 加载深度图像并进行处理
def process_depth_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    # 调整尺寸并重塑为3D数组
    resized_depth = cv2.resize(img, (size_2, size_1), interpolation=cv2.INTER_LINEAR).reshape((size_1, size_2, 1))
    return resized_depth


def load_all_img(sub_names, obj_names, episode_start, number, grasping_period):
    kinect_index = 1
    kinect_dir_path_list = []
    rgb_img_path_list = []
    depth_img_path_list = []
    traj_end_list = []
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
                        qpose_array = torch.load(qpose_path).cpu().numpy()
                        if not grasping_period:
                            traj_end = grasp_begin_index
                        else:
                            traj_end = qpose_array.shape[0]
                        episode_count += traj_end
                        number += 1
                        traj_end_list.append(traj_end)
                        kinect_folder_path = os.path.join(obj_dir, folder_name, "kinect", str(kinect_index))
                        kinect_dir_path_list.append(kinect_folder_path)
    
    for kinect_dir_path, traj_end in tqdm(zip(kinect_dir_path_list, traj_end_list), desc="Loading img path data", total=len(kinect_dir_path_list)):
        rgb_path_list_one, depth_path_list_one = load_one_folder_rgbd_path_list(kinect_dir_path, traj_end)
        rgb_img_path_list.extend(rgb_path_list_one)
        depth_img_path_list.extend(depth_path_list_one)
    # import pdb; pdb.set_trace()
    rgb_img_list = []
    depth_img_list = []
    
    with ThreadPoolExecutor(max_workers=12) as executor:
        rgb_img_list = list(tqdm(executor.map(process_rgb_image, rgb_img_path_list), total=len(rgb_img_path_list), desc="Loading and processing rgb img data"))
        depth_img_list = list(tqdm(executor.map(process_depth_image, depth_img_path_list), total=len(depth_img_path_list), desc="Loading and processing depth img data"))
    return [rgb_img_list, depth_img_list], episode_count, number


if __name__ == "__main__":
    train_data, train_episode_end, train_num = load_all_img(train_sub_names, train_obj, 0, 0, grasping_period)
    val_data_1, val_episode_end_1, val_num_1 = load_all_img(val_name_1, val_obj_1, train_episode_end, 0, grasping_period)
    val_data_2, val_episode_end_2, val_num_2 = load_all_img(val_name_2, val_obj_2, val_episode_end_1, val_num_1, grasping_period)
    val_data_3, val_episode_end_3, val_num_3 = load_all_img(val_name_3, val_obj_3, val_episode_end_2, val_num_2, grasping_period)
    test_data_1, test_episode_end_1, test_num_1 = load_all_img(test_name_1, test_obj_1, val_episode_end_3, 0, grasping_period)
    test_data_2, test_episode_end_2, test_num_2 = load_all_img(test_name_2, test_obj_2, test_episode_end_1, test_num_1, grasping_period)
    test_data_3, test_episode_end_3, test_num_3 = load_all_img(test_name_3, test_obj_3, test_episode_end_2, test_num_2, grasping_period)
    
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
    
    merge_rgb_array = np.array(train_data[0] + val_data_1[0] + val_data_2[0] + val_data_3[0] + test_data_1[0] + test_data_2[0] + test_data_3[0])
    merge_depth_array = np.array(train_data[1] + val_data_1[1] + val_data_2[1] + val_data_3[1] + test_data_1[1] + test_data_2[1] + test_data_3[1])

    print(f"merge_rgb_array.shape: {merge_rgb_array.shape}")
    print(f"merge_depth_array.shape: {merge_depth_array.shape}")
    
    # save the data
    np.save(f"{merge_dir}/image.npy", merge_rgb_array)
    np.save(f"{merge_dir}/depth.npy", merge_depth_array)
    
    
    