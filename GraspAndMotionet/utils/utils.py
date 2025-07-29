import torch
import numpy as np
import os
import json
import cv2
import yaml
from os.path import join as pjoin
from tqdm import tqdm
from pathlib import Path
from pytorch3d.transforms import rotation_6d_to_matrix,matrix_to_axis_angle
import random
import trimesh as tm


def size_splits(tensor, split_sizes, dim=0):
    """Splits the tensor according to chunks of split_sizes.

    Arguments:
        tensor (Tensor): tensor to split.
        split_sizes (list(int)): sizes of chunks
        dim (int): dimension along which to split the tensor.
    """
    if dim < 0:
        dim += tensor.dim()

    dim_size = tensor.size(dim)
    if dim_size != torch.sum(torch.Tensor(split_sizes)):
        raise KeyError("Sum of split sizes exceeds tensor dim")

    splits = torch.cumsum(torch.Tensor([0] + split_sizes), dim=0)[:-1]

    return tuple(tensor.narrow(int(dim), int(start), int(length))
                 for start, length in zip(splits, split_sizes))


def safe_search_file(root_path, pattern):
    """
    safely execute glob operation, skip files and directories that cannot be accessed
    Args:
        path (Path): the root path to search
        pattern (str): glob pattern to search for
    
    Returns:
        list: a list of Path objects that match the pattern
    """
    if type(root_path) is not Path:
        root_path = Path(root_path)
    result = []

    try:
        # 使用rglob代替glob来进行递归搜索
        for path_index,item in tqdm(enumerate(root_path.rglob(pattern)),"searching files"):
            try:
                # 尝试检查文件/目录是否存在和可访问
                if item.exists():
                    result.append(item)
            except (OSError, PermissionError) as e:
                print(f"error {item}: {str(e)}")
                continue

                
    except (OSError, PermissionError) as e:
        print(f"warning:  {root_path} error : {str(e)}")
    valid_paths = [p for p in result if p is not None]
    return valid_paths



def turn_hand_full_qpose_6d_to_axis_angle(hand_full_qpose):
    '''
    hand_full_qpose:
        has a batchsize 
        [batchsize,3 + 6 + 24]

    '''
    hand_6d_rot = hand_full_qpose[:,3:9]
    hand_axis_angle = matrix_to_axis_angle(rotation_6d_to_matrix(hand_6d_rot))
    return torch.cat([hand_full_qpose[:,:3],hand_axis_angle,hand_full_qpose[:,9:]],dim=1)






def resample_pcd_indices(pcd,random_sample_low_num):
    '''
    pcd: torch.tensor [num_points,3]
    random_sample_low_num: int, the number of points to sample
    '''
    assert pcd.shape[0] >= random_sample_low_num , "the number of points to sample should be less than the total number of points"
    sample_point_num = torch.randint(random_sample_low_num,pcd.shape[0],(1,)).item()
    resample_indices = np.random.permutation(pcd.shape[0])[:sample_point_num]
    # sample_pcd = pcd[resample_indices]

    return resample_indices



def from_vertices_to_trimesh_pcd(vertices):
    pcd = tm.PointCloud(vertices)
    return pcd

