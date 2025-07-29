
import open3d as o3d
import os
from os.path import join as pjoin
from time import sleep
from pprint import pprint
import numpy as np
import argparse

def arg_init():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--motion_index", default=0, type=int, help="motion_index"
    )

    args = parser.parse_args()
    return args



def main():
    arg = arg_init()
    motion_index = arg.motion_index
    #-------------------------------vis setting
    kinect_camera_view_parameter_path = "./all_data_global_view.json"
    kinect_camera_params = o3d.io.read_pinhole_camera_parameters(kinect_camera_view_parameter_path)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    #-------------------------------viz



    data_dir_path = pjoin(f"./viz",str(motion_index))
    obj_pcd_dir_path = pjoin(data_dir_path,"obj_pcd")
    predict_hand_mesh_dir_path = pjoin(data_dir_path,"model_predict_hand_mesh")
    final_qpose_dir_path = pjoin(data_dir_path,"final_qpose_mesh")
    gt_hand_mesh_dir_path = pjoin(data_dir_path,"gt_hand_mesh")

    predict_hand_mesh_num = os.listdir(predict_hand_mesh_dir_path)

    for idx in np.arange(len(predict_hand_mesh_num)):
        print(idx)
        sleep(0.01)
        predict_hand_mesh_path = pjoin(predict_hand_mesh_dir_path,f"hand_mesh_{idx}.ply")
        gt_hand_mesh_path = pjoin(gt_hand_mesh_dir_path,f"gt_hand_mesh_{idx}.ply")
        obj_pcd_path = pjoin(obj_pcd_dir_path,f"obj_pcd_{idx}.ply")
        final_qpose_path = pjoin(final_qpose_dir_path,f"final_pose_hand_mesh_{idx}.ply")

        predict_hand_mesh = o3d.io.read_triangle_mesh(predict_hand_mesh_path)
        predict_hand_mesh.compute_vertex_normals()
        gt_hand_mesh = o3d.io.read_triangle_mesh(gt_hand_mesh_path)
        gt_hand_mesh.compute_vertex_normals()
        obj_pcd = o3d.io.read_point_cloud(obj_pcd_path)
        final_qpose = o3d.io.read_triangle_mesh(final_qpose_path)

        vis.clear_geometries()
        vis.add_geometry(predict_hand_mesh)
        vis.add_geometry(gt_hand_mesh)
        vis.add_geometry(obj_pcd)
        vis.get_render_option().point_size = 5.0  
        vis.get_view_control().convert_from_pinhole_camera_parameters(kinect_camera_params)
        vis.poll_events()
        vis.update_renderer()
        vis.run()







if __name__ == "__main__":
    main()