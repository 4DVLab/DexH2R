import os
import sys
from os.path import join as pjoin
import open3d as o3d
sys.path.append(os.getcwd())



def save_camera_parameters(vis):
    '''
    used to registration the pick parameter key
    you have to point out the direct json save path
    '''
    parameters = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(
        camera_view_parameter_path, parameters)



def pick_pinhole_parameter(point_cloud_path,output_camera_view_parameter_path,is_mesh = False):
    '''
    parameter:
        input_camera_view_parameter_path: must be the json
    '''
    global camera_view_parameter_path 
    camera_view_parameter_path = output_camera_view_parameter_path
    if is_mesh:
        pcd = o3d.io.read_triangle_mesh(point_cloud_path)
        pcd.compute_vertex_normals()
    else:
        pcd = o3d.io.read_point_cloud(point_cloud_path)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.add_geometry(pcd)
    # 81 is the key code for "Q"
    vis.register_key_callback(81, save_camera_parameters)
    vis.run()
    vis.destroy_window()

pcd_view_path=  pjoin("./","obj_pcd_1.ply")
camera_view_output_file_path = pjoin("./","all_data_global_view.json")  

pick_pinhole_parameter(pcd_view_path,camera_view_output_file_path,is_mesh=False)

