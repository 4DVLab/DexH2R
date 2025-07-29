import os
os.chdir(os.getcwd())
import sys
sys.path.append(os.getcwd())
import torch
from utils.e3m5_hand_model import get_e3m5_handmodel
import mano
import numpy as np
from pytorch3d.structures import Meshes
from scipy.spatial import ConvexHull
import open3d as o3d
from tqdm import tqdm
from pathlib import Path
import argparse

def get_mano_vertices(mano_param: torch.Tensor, hand_type: str, return_faces: bool = False):
    """
    Return the vertices of mano mesh from mano_param(torch.tensor)

    :param mano_param: torch.tensor, shape=(batch_size, 61)
    :param hand_type: str, "left" / "right"
    :return vertices: torch.tensor, the vertices of mano mesh, shape=(batch_size, 778, 3)
    :return faces: (optional) np.ndarray, the faces of mano mesh, shape=(1538, 3)
    """
    
    mano_model_dir = "assets/mano_model"

    n_comps = 45
    batch_size = mano_param.shape[0]
    is_right = True if hand_type == "right" else False

    model_path = os.path.join(mano_model_dir, f"MANO_{hand_type.upper()}.pkl")


    beta = mano_param[:, 51:]
    global_r = mano_param[:, 3:6]
    pose = mano_param[:, 6:51]
    global_t = mano_param[:, :3]
    device = mano_param.device

    rh_model = mano.load(model_path=model_path,
                            is_right=is_right,
                            num_pca_comps=n_comps,
                            batch_size=batch_size,
                            flat_hand_mean=True).to(device)

    output = rh_model(betas=beta,
                        global_orient=global_r,
                        hand_pose=pose,
                        transl=global_t,
                        return_verts=True,
                        return_tips=True)

    result = (output.vertices, rh_model.faces) if return_faces else output.vertices
    return result




def is_point_in_bbox(points, bbox):
    """
    判断每个点是否在给定的边界框内。

    参数:
        points (torch.Tensor): 点集，形状为 (batch_size, point_num, 3)。
        bbox (torch.Tensor): 边界框，形状为 (2, 3) bbox[0]是最小值点 bbox[1]是最大值点。

    返回:
        mask (torch.Tensor): 布尔张量，形状为 (batch_size, point_num)，表示每个点是否在边界框内。
    """
    # 扩展 bbox 的形状以匹配 points 的形状
    bbox = bbox.unsqueeze(0).unsqueeze(0)  # 形状变为 (1, 1, 2, 3)

    # 比较每个点的坐标是否在 bbox 的最小值和最大值之间
    min_vals = bbox[:, :, 0, :]  # 最小值点，形状为 (1, 1, 3)
    max_vals = bbox[:, :, 1, :]  # 最大值点，形状为 (1, 1, 3)

    # 判断每个点是否在边界框内
    mask = (points >= min_vals) & (points <= max_vals)  # 形状为 (batch_size, point_num, 3)
    mask = mask.all(dim=-1)  # 在所有维度上满足条件，形状为 (batch_size, point_num)

    return mask


def filter_mano_collision_with_mano_boundingbox_and_completion(root_path, hand_type, device, final_pose_file_name):
    print(root_path)
    mano_path = os.path.join(root_path, f"{hand_type}_mano.pt")
    mano_data = torch.load(mano_path).to(device).to(torch.float32)    # (batch_size, 61)
    mano_vertices = get_mano_vertices(mano_data, hand_type=hand_type)  # (batch_size, 778, 3)

    # Calculate mano bounding box
    min_vals, _ = torch.min(mano_vertices, dim=1)  # 形状为 (batch_size, 3)
    max_vals, _ = torch.max(mano_vertices, dim=1)  # 形状为 (batch_size, 3)
    bounding_boxes = torch.stack([min_vals, max_vals], dim=1)  # 形状为 (batch_size, 2, 3)

    qpose_path = os.path.join(root_path, f"{final_pose_file_name}.pt")
    new_qpose_path = os.path.join(root_path, f"{final_pose_file_name}_mano_filter_completion.pt")
    qpose_data = torch.load(qpose_path).to(device).to(torch.float32)  # (batch_size, 500, 30)

    hand_model = get_e3m5_handmodel(remove_wrist=True, device=device)
    # qpose_data_batch = qpose_data.reshape(-1, qpose_data.shape[2])      # (-1, 30)

    # Process one batch qpose (500, 3)
    for idx in range(mano_data.shape[0]):
        qpose_data_one_batch = qpose_data[idx, :, :]
        hand_pcd, hand_faces = hand_model.get_meshes_from_q(qpose_data_one_batch, batch_mode=True)
        batch_hand_pcd = torch.cat(hand_pcd, dim=1)         # hand_pcd: (500, shadowhand_points, 3)

        mano_bounding_box = bounding_boxes[idx, :, :]
        mask = is_point_in_bbox(batch_hand_pcd, mano_bounding_box)  # (500, shadowhand_points)
        collision_result = torch.any(mask)

        if collision_result:
            # 只留下valid的qpose
            new_qpose_data_mask = torch.sum(mask, dim=1) == 0

            invalid_num = 500-sum(new_qpose_data_mask).item()
            print(idx, invalid_num)

            # (500-invalid_num, 30)
            new_qpose_data_one_batch = qpose_data_one_batch[new_qpose_data_mask, :]

            # Completion
            if invalid_num > 490:
                complete_num = invalid_num - 490
                valid_indices = torch.randperm(500)[:complete_num]
                complete_qpose = qpose_data[idx, valid_indices]

                new_qpose_data_one_batch = torch.cat((new_qpose_data_one_batch, complete_qpose), dim=0)
                # qpose_data[idx, 10-complete_num:10] = qpose_data[idx, valid_indices]

            # 清空当前batch_idx的qpose
            qpose_data[idx, :, :] = torch.zeros((500, 30))

            # 赋值
            valid_qpose_num = new_qpose_data_one_batch.shape[0]
            qpose_data[idx, :valid_qpose_num, :] = new_qpose_data_one_batch

    torch.save(qpose_data, new_qpose_path)


def gen_path_file(root_path, txt_path):
    '''
    param:
        root_path: str
    '''
    mano_file_path_list = Path(root_path).rglob("*mano.pt")
    mano_configuration_file_list = [[str(mano_file_path.parent), mano_file_path.stem.split("_")[0]] for mano_file_path in mano_file_path_list]
    with open(txt_path, 'w', encoding='utf-8') as f:
        for item in mano_configuration_file_list:
            f.write(' '.join(item) + '\n')


def arg_init():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--final_pose_file_name", default="model_final_qpose_cvae", help="model_final_qpose_cvae or model_final_qpose_dexgraspanything"
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    device = "cuda" 
    
    datset_root_path = "./../dataset/DexH2R_dataset"
    txt_path = "mano_collision/root_path_mano_data.txt"
    gen_path_file(datset_root_path, txt_path)

    args = arg_init()
    final_pose_file_name = args.final_pose_file_name
    
    root_path_data = []
    with open(txt_path, "r") as file:
        for line in file:
            root_path_data.append(line.strip().split())

    for root_path, hand_type in tqdm(root_path_data, file=sys.stderr):
        filter_mano_collision_with_mano_boundingbox_and_completion(root_path, hand_type, device, final_pose_file_name)
