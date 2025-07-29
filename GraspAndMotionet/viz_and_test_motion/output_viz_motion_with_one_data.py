import os
import torch
import sys
sys.path.append(os.getcwd())
import numpy as np
from os.path import join as pjoin
from tqdm import tqdm
from utils.utils import from_vertices_to_trimesh_pcd
from utils.e3m5_hand_model import get_e3m5_handmodel
from os.path import join as pjoin
import trimesh as tm
from tqdm import tqdm
from os.path import join as pjoin
import open3d as o3d
import argparse

# 传入一个路径，可视化这个路径的文件内容





def create_hand_mesh(point_list,face_list,o3d_flag = False):
    mesh_list = []
    for point,face in zip(point_list,face_list):
        if o3d_flag:
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(point.cpu().numpy())
            mesh.triangles = o3d.utility.Vector3iVector(face.cpu().numpy())
            mesh.compute_vertex_normals()
            mesh_list.append(mesh)
        else:
            mesh = tm.Trimesh(vertices=point.cpu().numpy(),faces=face.cpu().numpy())
            mesh_list.append(mesh)
    
    return mesh_list


def load_model():
    model_save_dir_path = "./../dataset/object_model/"
    model_dict = {}
    for file_name in os.listdir(model_save_dir_path):
        if file_name.endswith(".obj"):
            model_dict[file_name[:-4]] = tm.load(pjoin(model_save_dir_path,file_name))
    return model_dict



# add_ padding
def gen_one_viz_data(model_predict_data,data_viz_index,output_all_dir_path,model_dict,hand_model):

    # make_save_dir start
    output_dir_path = pjoin(output_all_dir_path,str(data_viz_index))
    model_predict_hand_mesh_dir_path = pjoin(output_dir_path,"model_predict_hand_mesh")
    obj_pcd_save_dir_path = pjoin(output_dir_path,"obj_pcd")
    obj_mesh_pcd_svae_dir_path = pjoin(output_dir_path,"obj_mesh_pcd")
    gt_hand_save_mesh_dir_path = pjoin(output_dir_path,"gt_hand_mesh")
    final_qpose_save_mesh_dir_path = pjoin(output_dir_path,"final_qpose_mesh")
    every_frame_final_pose_save_mesh_dir_path = pjoin(output_dir_path,"every_frame_final_pose_mesh")

    os.makedirs(obj_mesh_pcd_svae_dir_path,exist_ok=True)
    os.makedirs(final_qpose_save_mesh_dir_path,exist_ok=True)
    os.makedirs(gt_hand_save_mesh_dir_path,exist_ok=True)
    os.makedirs(obj_pcd_save_dir_path,exist_ok=True)
    os.makedirs(model_predict_hand_mesh_dir_path,exist_ok=True)
    os.makedirs(every_frame_final_pose_save_mesh_dir_path,exist_ok=True)
    # end



    gt_data_dir_path = model_predict_data[data_viz_index]["gtdata_dir_path"]
    sample_model_predict_data = model_predict_data[data_viz_index]

    model_name = gt_data_dir_path.split('/')[-2]

    model_predict_hand = sample_model_predict_data["model_predict_traj_tensor"].to("cuda")

    hand_pcd,hand_faces = hand_model.get_meshes_from_q(model_predict_hand,batch_mode=True)
    for idx in tqdm(np.arange(hand_pcd[0].shape[0]),desc = "model_predict_hand_mesh"):

        hand_mesh_list = create_hand_mesh([pcd[idx] for pcd in hand_pcd],hand_faces,o3d_flag=True)
        hand_mesh = o3d.geometry.TriangleMesh()
        for mesh in hand_mesh_list:
            hand_mesh += mesh
        # hand_mesh = tm.util.concatenate(hand_mesh_list)
        if idx >= 5:
            frame_flag = sample_model_predict_data["infer_frame_flag_list"][idx]
            if frame_flag:
                hand_mesh.paint_uniform_color([1,0,0])
            else:
                hand_mesh.paint_uniform_color([0,1,0])
        o3d.io.write_triangle_mesh(pjoin(model_predict_hand_mesh_dir_path,f"hand_mesh_{idx}.ply"),hand_mesh)



    obj_mesh = model_dict[model_name]
    obj_pose_path = pjoin(gt_data_dir_path,"obj_pose.pt")
    obj_pose = torch.load(obj_pose_path)
    for data_idx in tqdm(np.arange(obj_pose.shape[0]),desc = "obj_pcd"):
        obj_mesh_temp = obj_mesh.copy().apply_transform(obj_pose[data_idx])
        obj_mesh_temp.export(pjoin(obj_mesh_pcd_svae_dir_path,f"obj_pcd_{data_idx}.ply"))
        if data_idx == obj_pose.shape[0] - 1:
            for idx in np.arange(obj_pose.shape[0],obj_pose.shape[0] + 100):
                obj_mesh_temp.export(pjoin(obj_mesh_pcd_svae_dir_path,f"obj_pcd_{idx}.ply"))


    obj_pcd_path = pjoin(gt_data_dir_path,"real_obj_pcd_xyz.pt")
    obj_pcd = torch.load(obj_pcd_path)
    for data_idx in tqdm(np.arange(obj_pcd.shape[0]),desc = "obj_pcd"):
        pcd = from_vertices_to_trimesh_pcd(obj_pcd[data_idx])
        pcd.export(pjoin(obj_pcd_save_dir_path,f"obj_pcd_{data_idx}.ply"))
        if data_idx == obj_pcd.shape[0] - 1:
            for idx in np.arange(obj_pcd.shape[0],obj_pcd.shape[0] + 100):
                pcd.export(pjoin(obj_pcd_save_dir_path,f"obj_pcd_{idx}.ply"))



    gt_hand_data_dir_path = pjoin(gt_data_dir_path,"qpos.pt")
    gt_hand_data = torch.load(gt_hand_data_dir_path).to("cuda").to(torch.float32)
    hand_pcd,hand_faces = hand_model.get_meshes_from_q(gt_hand_data,batch_mode=True)
    for idx in tqdm(np.arange(hand_pcd[0].shape[0]),desc = "gt_hand_mesh"):
        hand_mesh_list = create_hand_mesh([pcd[idx] for pcd in hand_pcd],hand_faces,o3d_flag=True)
        hand_mesh = o3d.geometry.TriangleMesh()
        for mesh in hand_mesh_list:
            hand_mesh += mesh
        hand_mesh.paint_uniform_color([0,0,1])
        o3d.io.write_triangle_mesh(pjoin(gt_hand_save_mesh_dir_path,f"gt_hand_mesh_{idx}.ply"),hand_mesh)


    if type(sample_model_predict_data["model_select_final_pose"]) == list:
        final_qpose_data = torch.stack(sample_model_predict_data["model_select_final_pose"]).to("cuda")
    else:
        final_qpose_data = sample_model_predict_data["model_select_final_pose"].to("cuda")

    hand_pcd,hand_faces = hand_model.get_meshes_from_q(final_qpose_data,batch_mode=True)
    for idx in tqdm(np.arange(hand_pcd[0].shape[0]),desc = "final_pose_hand_mesh"):
        hand_mesh_list = create_hand_mesh([pcd[idx] for pcd in hand_pcd],hand_faces,o3d_flag=True)
        hand_mesh = o3d.geometry.TriangleMesh()
        for mesh in hand_mesh_list:
            hand_mesh += mesh   
        hand_mesh.paint_uniform_color([0.5,0.5,0.5])
        hand_mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(pjoin(final_qpose_save_mesh_dir_path,f"final_pose_hand_mesh_{idx}.ply"),hand_mesh)


    if type(sample_model_predict_data["every_frame_final_pose"]) == list:
        final_qpose_data = torch.stack(sample_model_predict_data["every_frame_final_pose"]).to("cuda")
    else:
        final_qpose_data = sample_model_predict_data["every_frame_final_pose"].to("cuda")

    hand_pcd,hand_faces = hand_model.get_meshes_from_q(final_qpose_data,batch_mode=True)
    for idx in tqdm(np.arange(hand_pcd[0].shape[0]),desc = "every_frame_final_pose_hand_mesh"):
        hand_mesh_list = create_hand_mesh([pcd[idx] for pcd in hand_pcd],hand_faces,o3d_flag=True)
        hand_mesh = o3d.geometry.TriangleMesh()
        for mesh in hand_mesh_list:
            hand_mesh += mesh   
        hand_mesh.paint_uniform_color([0.5,0.5,0.5])
        hand_mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(pjoin(every_frame_final_pose_save_mesh_dir_path,f"every_frame_final_pose_hand_mesh_{idx}.ply"),hand_mesh)




def gen_viz_data(data_viz_indices:list,traj_data_path):

    output_all_dir_path = pjoin(os.path.dirname(traj_data_path),f"viz")
    os.makedirs(output_all_dir_path,exist_ok=True)
    model_predict_data = torch.load(traj_data_path)
    hand_model = get_e3m5_handmodel(remove_wrist = True,device= "cuda")
    model_dict = load_model()

    for data_viz_index in tqdm(data_viz_indices,desc = "gen_viz_data"):
        gen_one_viz_data(model_predict_data,data_viz_index,output_all_dir_path,model_dict,hand_model)


def arg_init():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir_name", default = "test", help="the dir name of the data you want to evaluate, test/ablation."
    )
    parser.add_argument('--data_seq_indices', nargs='+', type=int, help='save the mesh, pcd and so on... in the inference data, so you can vizualize it')

    args = parser.parse_args()
    return args



def main():
    args = arg_init()
    data_path = f"./viz_motion_output/{args.dir_name}/vis_data.pth"
    gen_viz_data(args.data_seq_indices,data_path)


if __name__ == "__main__":
    main()