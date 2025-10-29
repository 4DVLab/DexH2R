import os
import sys
sys.path.append(os.getcwd())

import argparse
import colorama
import numpy as np
import open3d as o3d
import torch
import copy
import cv2
import mano
from os.path import join as pjoin
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from utils.github_e3m5_hand_model import get_e3m5_handmodel
# from train_data_process.hand_arm_model.hand_arm_model import get_hand_arm_model
import json
from scipy.spatial.transform import Rotation
from pathlib import Path

def load_json_param(param_path):
    if not os.path.exists(param_path):
        raise FileNotFoundError(f"param file {param_path} not found")
    with open(param_path,"r") as file_handler:
        param = json.load(file_handler)
    return param


def load_camera_intrinsics(folder_path, json_name="camera_intrinsics.json"):
    '''
    intrinsics keys:
        width int
        height int
        camera_intrinsics np.array 3x3
        distortion_parameter 8x1
    
    '''
    intrinsics_file_path =pjoin(folder_path, json_name)
    intrinsics = load_json_param(intrinsics_file_path)
    intrinsics["matrix"] = np.array(intrinsics["camera_intrinsics"]).reshape((3,3))
    intrinsics["distortion"] = np.array(intrinsics["distortion_parameter"]).reshape((-1,1))

    return intrinsics

def load_internal_camera_extrinsics(cam_extrinsics_folder: Path, cam_num = None,another_file_name = None):
    '''
    return:
        extrinsics:(rotation: x,y,z,w; translation: x,y,z),matrix (4,4)
    '''
    
    if another_file_name is not None:
        cam_extrinsics_path = Path(cam_extrinsics_folder) / another_file_name
    else:
        cam_extrinsics_path = Path(cam_extrinsics_folder) / Path(f"cali0{cam_num}.json")
    extrinsics = None
    with open(cam_extrinsics_path, "r") as json_reader:
        camera_data = json.load(json_reader)
        quat_order = ["x", "y", "z", "w"]
        rotation = np.array(list([camera_data["value0"]["rotation"][key] for key in quat_order])).flatten()
        translation_order = ["x", "y", "z"]
        translation = np.array(
            list([camera_data["value0"]["translation"][key] for key in translation_order])).flatten()
        extrinsics = seven_num2matrix(translation, rotation)
    return extrinsics

def seven_num2matrix(translation,roatation):#
    '''
    parameter:
        translation x,y,z rotation x,y,z,w
    return:
        numpy array (4x4)
    '''
    transform_matrix = np.identity(4)
    transform_matrix[:3,:3] = Rotation.from_quat(roatation).as_matrix()
    transform_matrix[:3,3] = translation
    return transform_matrix


def create_point_cloud_from_rgb_and_depth(undistorted_rgb: np.ndarray, 
                                            undistorted_depth: np.ndarray, 
                                            camera_inrisics: np.ndarray, 
                                            width: int, height: int) -> o3d.geometry.PointCloud:
    """
    rgb+depth(np.ndarray) -> rgb+depth(o3d) -> rgbd(o3d) -> pcd(o3d)

    :param undistorted_rgb: np.ndarray
    :param undistorted_depth: np.ndarray
    :param camera_inrisics: np.ndarray((3, 3))
    :param width: int
    :param height: int
    :return pcd: o3d.geometry.PointCloud
    """
    # Convert undistorted images to open3d format
    color_raw = o3d.geometry.Image(undistorted_rgb)
    depth_raw = o3d.geometry.Image(undistorted_depth)

    # Create an RGBD image (mm->m)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False)

    # Convert to point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            width, height, 
            camera_inrisics[0, 0], camera_inrisics[1, 1], 
            camera_inrisics[0, 2], camera_inrisics[1, 2]
        )
    )

    # Return the point cloud
    return pcd


class hand_arm_data():
    def __init__(self, root_path) -> None:
        '''
        init hand_arm_data

        :param data_index: the name of sequencd
        :param frame_delay_index: hand_arm_begin_index from sync
        :param _data_cache_folder_path: root_path of the 4 minutes sequence
        '''
        self.root_path = root_path

        qpos_path = pjoin(self.root_path, "qpos.pt")
        # qpos_path = pjoin(self.root_path, "hand_arm_full_qpos.pt")
        self.qpos_data = torch.load(qpos_path).float().to("cuda")

        self.e3m5_hand_model = get_e3m5_handmodel("cuda")
        # self.e3m5_hand_model = get_hand_arm_model("cuda")

    def get_hand_arm_mesh(self, frame_idx):
        """
        get the mesh of hand_arm whose timestamp is the closest to time_stamp

        :param time_stamp: the current timestamp
        :return hand_arm_mesh: o3d mesh
        """
        hand_arm_mesh = self.e3m5_hand_model.get_meshes_from_q(
            q=self.qpos_data[frame_idx].unsqueeze(0),
            type=1)

        # # 手动添加world到ra_base_link的变换
        # world_to_ra_base_transform = np.array([
        #     [1.0, 0.0, 0.0, 0.0],
        #     [0.0, 1.0, 0.0, 0.0], 
        #     [0.0, 0.0, 1.0, 0.75509999999999999343],
        #     [0.0, 0.0, 0.0, 1.0]
        # ])
        # hand_arm_mesh.transform(world_to_ra_base_transform)

        return hand_arm_mesh



class kinect_img():
    def __init__(self, root_path, cam_index) -> None:
        '''
        init kinect_data

        :param data_index: the name of sequencd
        :param _data_cache_folder_path: root_path of the 4 minutes sequence
        '''
        self.root_path = root_path
        self.kinect_data_folder_path = pjoin(self.root_path, "kinect")
        self.cam_index = cam_index
        self.rgb_data_folder_path = pjoin(self.root_path, "kinect", str(cam_index), "rgb")


        # Kinect intrinsics and extrinsics
        self.kinect_intrinsics = None
        self.load_kinect_intriscis()

        hand_arm_to_kinect_pcd_0_transform = load_internal_camera_extrinsics(
            "/media/lab4dv/python/camera_calibration/camera_extrinsics", 
            another_file_name="hand_arm_mesh_to_kinect_pcd_0.json"
        )
        self.kinect_pcd_0_to_hand_arm_transform = np.linalg.inv(hand_arm_to_kinect_pcd_0_transform)

        self.kinect_extrinsics_list = []
        self.load_kinect_extrinsics()

        self.merged_pcd_list = None

        self.img_num = len(os.listdir(self.rgb_data_folder_path))

        print("load kinect data ready, you can use get_camera_pcd_list to get the pcd list")

    def load_kinect_intriscis(self, dir_path="/media/lab4dv/python/camera_calibration/camera_intrinsics/kinect"):
        json_name = f"camera_intrinsics_{self.cam_index}.json"
        self.kinect_intrinsics = load_camera_intrinsics(dir_path, json_name)

    def load_kinect_extrinsics(self):
        """
        Load the extrinsics of the kinect cameras
        """
        extrinsics = load_internal_camera_extrinsics(
            "/media/lab4dv/python/camera_calibration/camera_extrinsics/kinect", 
            cam_num=self.cam_index
        )
        extrinsics = self.kinect_pcd_0_to_hand_arm_transform @ extrinsics
        self.kinect_extrinsics = extrinsics

    def project_hand_arm_mesh_to_img(self, hand_arm_mesh):
        """
        Project the hand_arm_mesh to the kinect image
        :param hand_arm_mesh: o3d.geometry.TriangleMesh
        :return: o3d.geometry.PointCloud
        """
        hand_arm_mesh.transform(np.linalg.inv(self.kinect_extrinsics))

        return hand_arm_mesh


    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """
        Rectify the distortion of one iamge

        :param image: np.ndarray((3, 3))
        :return undistorted_image: the image that is rectifed distortion
        """
        camera_matrix = self.kinect_intrinsics["matrix"]
        distortion_coeffs = self.kinect_intrinsics["distortion"]

        h, w = image.shape[:2]

        # Rectify distortion
        mapx, mapy = cv2.initUndistortRectifyMap(
            camera_matrix, distortion_coeffs, None, None, (w, h), 5)
        undistorted_image = cv2.remap(
            image, mapx, mapy, interpolation=cv2.INTER_NEAREST)

        return undistorted_image

    def get_one_frame_rgb(self, frame_idx) -> np.ndarray:
        """
        undistort rgb & depth -> pcd -> pcd (in global)
        """
        rgb = cv2.imread(pjoin(self.rgb_data_folder_path, f"{frame_idx}.jpg"))
        undistorted_rgb = self.undistort_image(rgb)
        return undistorted_rgb

def set_vizulization_argparse(parser):
    """
    init set visualiation argparse

    :param hand_arm_viz: True if it exists
    :param kinect_viz: True if it exists
    :param realsense_viz: True if it exists
    :param mano_viz: True if it exists
    """
    parser.add_argument("--root_path", type=str,
                        help="the root_path of one handover sequence",
                        required=True
    )



def init_and_get_args():
    """
    call set_vizulization_argparse(parser)

    :param data_index: default=0
    :param hand_arm_viz: True if it exists
    :param kinect_viz: True if it exists
    :param realsense_viz: True if it exists
    :param mano_viz: True if it exists
    """
    colorama.init(autoreset=True)
    parser = argparse.ArgumentParser()
    set_vizulization_argparse(parser)
    args = parser.parse_args()

    return args


def create_overlay_image(capture_image, kinect_image, alpha=0.5):
    """
    将capture的图像和kinect图像按比例相加，创建重叠图像
    
    :param capture_image: 从可视化capture的图像 (numpy array)
    :param kinect_image: kinect原始图像 (numpy array)
    :param alpha: capture图像的权重 (0-1)，kinect图像权重为(1-alpha)
    :return: 重叠后的图像
    """
    # 确保两个图像尺寸一致
    if capture_image.shape != kinect_image.shape:
        # 调整kinect图像尺寸以匹配capture图像
        print("size is not match")
        kinect_image = cv2.resize(kinect_image, (capture_image.shape[1], capture_image.shape[0]))
    
    # 将图像转换为float32进行混合
    capture_float = capture_image.astype(np.float32)
    kinect_float = kinect_image.astype(np.float32)
    
    # 按比例混合图像
    overlay_image = alpha * capture_float + (1 - alpha) * kinect_float
    
    # 转换回uint8
    overlay_image = np.clip(overlay_image, 0, 255).astype(np.uint8)
    
    return overlay_image


def main():
    args = init_and_get_args()

    root_path = args.root_path    # "./data_cache"

    kinect_img_data_handler = kinect_img(root_path, cam_index = 1)
    seq_length = kinect_img_data_handler.img_num
    hand_arm_data_handler = hand_arm_data(root_path)

    #-------------------------------vis setting
    # kinect_camera_view_parameter_path = "visualize_dataset/vis_view_params.json"
    camera_params = o3d.camera.PinholeCameraParameters()

    kinect_camera_params = o3d.camera.PinholeCameraIntrinsic(
        width=kinect_img_data_handler.kinect_intrinsics["width"],
        height=kinect_img_data_handler.kinect_intrinsics["height"],
        fx=kinect_img_data_handler.kinect_intrinsics["matrix"][0, 0],
        fy=kinect_img_data_handler.kinect_intrinsics["matrix"][1, 1],
        cx=kinect_img_data_handler.kinect_intrinsics["matrix"][0, 2],
        cy=kinect_img_data_handler.kinect_intrinsics["matrix"][1, 2]
    )
    camera_params.intrinsic = kinect_camera_params
    camera_params.extrinsic = np.eye(4)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    first_vis_flag = True

    # 创建输出目录
    output_dir = "test_meshes/korea_guy_issue"
    overlay_dir = "test_meshes/korea_guy_issue_overlay"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    for index in range(seq_length):
        print(colorama.Fore.GREEN + f"current frame_idx: {index}")

        if first_vis_flag:
            first_vis_flag = False
        else:
            vis.clear_geometries()

        hand_arm_mesh = hand_arm_data_handler.get_hand_arm_mesh(index)
        hand_arm_mesh = kinect_img_data_handler.project_hand_arm_mesh_to_img(hand_arm_mesh)
        vis.add_geometry(hand_arm_mesh)

        vis.get_view_control().convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)
        vis.poll_events()
        vis.update_renderer()
        
        # Capture可视化图像
        capture_image = np.asarray(vis.capture_screen_float_buffer(False))
        capture_image = (capture_image * 255).astype(np.uint8)
        
        # 获取对应的kinect图像
        kinect_image = kinect_img_data_handler.get_one_frame_rgb(index)
        
        # 创建重叠图像
        overlay_image = create_overlay_image(capture_image, kinect_image, alpha=0.6)
        
        # 保存图像
        cv2.imwrite(f"{output_dir}/{index}.png", capture_image)
        cv2.imwrite(f"{overlay_dir}/{index}.png", overlay_image)
        
        print(f"Saved capture image: {output_dir}/{index}.png")
        print(f"Saved overlay image: {overlay_dir}/{index}.png")
        
        vis.run()


if __name__ == "__main__":
    main()
