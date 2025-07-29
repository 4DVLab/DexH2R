import open3d as o3d
import os
from tqdm import tqdm
from os.path import join as pjoin
import argparse
import numpy as np
import torch
from pytorch3d.ops import sample_farthest_points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--point_num', type=int, default = 4096, help='The number of points sampled from the mesh surface ')
    args = parser.parse_args()
    point_num = args.point_num

    dataset_folder_path = "./../dataset"

    model_data_dir_path = pjoin(dataset_folder_path, "object_model")
    output_dir_path = pjoin(dataset_folder_path, f"obj_surface_pcd_{point_num}")
    os.makedirs(output_dir_path,exist_ok=True)

    for file_name in tqdm(os.listdir(model_data_dir_path)):
        if file_name.endswith(".obj"):
            model_path = pjoin(model_data_dir_path, file_name) 
            obj_mesh_o3d = o3d.io.read_triangle_mesh(model_path)
            obj_mesh_o3d.compute_vertex_normals().orient_triangles()
            obj_surface_pcd_o3d = obj_mesh_o3d.sample_points_uniformly(number_of_points=20000)
            obj_surface_pcd_torch = torch.from_numpy(np.asarray(obj_surface_pcd_o3d.points)).float().to("cuda")
            obj_surface_pcd_normals = np.asarray(obj_surface_pcd_o3d.normals)
            obj_surface_uniform_pcd, obj_surface_sample_indices = sample_farthest_points(obj_surface_pcd_torch.unsqueeze(0),K=point_num)
            obj_surface_uniform_pcd_o3d = o3d.geometry.PointCloud()
            obj_surface_uniform_pcd_o3d.points = o3d.utility.Vector3dVector(obj_surface_uniform_pcd.squeeze(0).cpu().numpy())
            obj_surface_uniform_pcd_o3d.normals = o3d.utility.Vector3dVector(obj_surface_pcd_normals[obj_surface_sample_indices.squeeze(0).cpu().numpy()])

            pcd_save_path = pjoin(output_dir_path,file_name.split(".")[0] + ".ply")
            o3d.io.write_point_cloud(pcd_save_path,obj_surface_uniform_pcd_o3d)


if __name__ == "__main__":
    main()    