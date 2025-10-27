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
from utils.e3m5_hand_model import get_e3m5_handmodel
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


class mano_data():
    def __init__(self, root_path, mano_model_dir="visualize_dataset/mano_model"):
        '''
        init mano_Data

        :param root_path: handover sequence data root path
        :param mano_model_dir: the dir path of MANO_{LEFT/RIGHT}.pkl
        '''
        self.root_path = root_path

        # self.load_mano_to_global_transform()

        if os.path.exists(f"{root_path}/left_mano.pt"):
            self.hand_type = "left"
        else:
            self.hand_type = "right"
        self.mano_models = os.path.join(mano_model_dir, f"MANO_{self.hand_type.upper()}.pkl")

        mano_pt_path = f"{root_path}/{self.hand_type}_mano.pt"
        self.mano_params = torch.load(mano_pt_path)

    def load_mano_to_global_transform(self):
        """
            get RT from ZCAM to global (?)
            calibration/zcam_to_kinect_transform.json
        """
        calib_path = os.path.join(self.root_path, "calibration")
        mano_to_kinect_transform = load_internal_camera_extrinsics(calib_path,
            another_file_name="zcam_to_kinect_transform.json")
        global_to_kinect_transform = load_internal_camera_extrinsics(calib_path,
            another_file_name="hand_arm_mesh_to_kinect_pcd_0.json")
        self.mano_to_global_transform = np.linalg.inv(global_to_kinect_transform) @ mano_to_kinect_transform
    
    def get_mano_mesh(self, frame_idx):
        """
        get the one humanhand mesh on frame_idx

        :param frame_idx: current frame index
        :return humanhand_mesh:
        """

        n_comps = 45
        batch_size = 1

        mano_param = self.mano_params[frame_idx].float()
        beta = mano_param[51:].reshape(1, -1)
        global_r = mano_param[3:6].reshape(1, -1)  # (1,3)
        pose = mano_param[6:51].reshape(1, -1)
        global_t = mano_param[:3].reshape(1, -1)
        
        rh_model = mano.load(model_path=self.mano_models,
                        num_pca_comps=n_comps,
                        batch_size=batch_size,
                        flat_hand_mean=True)

        output = rh_model(betas=beta,
                    global_orient=global_r,
                    hand_pose=pose,
                    transl=global_t,
                    return_verts=True,
                    return_tips=True)
        
        vertices = output.vertices[0, :, :].numpy()
        faces = rh_model.faces
        
        # ZCAM -> global
        hand_mesh = o3d.geometry.TriangleMesh()
        hand_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        hand_mesh.triangles = o3d.utility.Vector3iVector(faces)
        # hand_mesh.transform(self.mano_to_global_transform)
        hand_mesh.compute_vertex_normals()

        return hand_mesh


class object_data():
    def __init__(self, root_path, obj_name, obj_mesh_folder="/home/lab4dv/DynamicGrasp/models") -> None:
        '''
        init kinect_data

        :param data_index: the name of sequencd
        :param frame_delay_index: kinect_begin_index from sync, must be the camera 0, not the camera 1/2/3
        :param _data_cache_folder_path: root_path of the 4 minutes sequence
        '''
        self.root_path = root_path
        self.obj_pose_pt_path = f"{root_path}/obj_pose.pt"
        self.obj_pose_data = torch.load(self.obj_pose_pt_path)

        self.obj_mesh_path = pjoin(obj_mesh_folder, f"{obj_name}.obj")
        self.obj_mesh = o3d.io.read_triangle_mesh(self.obj_mesh_path)

        # print("load object data ready, you can use get_object_mesh to get the geometry")

    def get_sequence_length(self):
        """
        Return the length of the object pose data
        :return: the length of the object pose data
        """
        return self.obj_pose_data.shape[0]

    def get_object_mesh(self, frame_idx, color=[0.0, 1.0, 0.0]):
        """
        Return the transformed object mesh at frame_idx
        :param frame_idx: the index of the frame to visualize
        :param color: the color to paint the object mesh
        :return: the transformed object mesh
        """
        obj_mesh = copy.deepcopy(self.obj_mesh)
        obj_mesh.transform(self.obj_pose_data[frame_idx].numpy())
        obj_mesh.compute_vertex_normals()
        obj_mesh.paint_uniform_color(color)

        return obj_mesh


class kinect_data():
    def __init__(self, root_path, without_crop=False) -> None:
        '''
        init kinect_data

        :param data_index: the name of sequencd
        :param _data_cache_folder_path: root_path of the 4 minutes sequence
        '''
        self.root_path = root_path
        self.kinect_data_folder_path = pjoin(self.root_path, "kinect")
        self.data_folder_path = [pjoin(self.root_path, "kinect", str(index)) for index in np.arange(4)]

        # Flag whether viz with no-cropped pcd, default is False 
        self.crop = not without_crop
        self.crop_bounding_box = [[0.359287, -0.747868, 0.87556], [1.64417, 0.864404, 1.87057]]

        # Kinect intrinsics and extrinsics
        self.kinect_intrinsics_list = []
        self.load_kinect_intriscis()

        hand_arm_to_kinect_pcd_0_transform = load_internal_camera_extrinsics(
            os.path.join(root_path, "calibration"), 
            another_file_name="hand_arm_mesh_to_kinect_pcd_0.json"
        )
        self.kinect_pcd_0_to_hand_arm_transform = np.linalg.inv(hand_arm_to_kinect_pcd_0_transform)

        self.kinect_extrinsics_list = []
        self.load_kinect_extrinsics()

        self.merged_pcd_list = None
        # self.get_merged_pcd_from_all_kinect()

        print("load kinect data ready, you can use get_camera_pcd_list to get the pcd list")

    def load_kinect_intriscis(self, dir_path="visualize_dataset/camera_intrinsics"):
        for i in range(4):
            json_name = f"kinect_{i}.json"
            self.kinect_intrinsics_list.append(load_camera_intrinsics(dir_path, json_name))

    def load_kinect_extrinsics(self):
        """
        Load the extrinsics of the kinect cameras
        """
        for i in range(4):
            extrinsics = load_internal_camera_extrinsics(
                os.path.join(self.root_path, "calibration/kinect"), 
                cam_num=i
            )
            extrinsics = self.kinect_pcd_0_to_hand_arm_transform @ extrinsics
            self.kinect_extrinsics_list.append(extrinsics)

    def get_merged_pcd_from_all_kinect(self):
        """
        Get all merged pcd with progress tracking
        """
        # Load all rgb and depth
        rgb_list = [torch.load(pjoin(self.root_path, f"kinect_{i}_rgb.pt")) for i in range(4)]
        depth_list = [torch.load(pjoin(self.root_path, f"kinect_{i}_depth.pt")) for i in range(4)]
        seq_length = rgb_list[0].shape[0]

        merged_pcd_list = [o3d.geometry.PointCloud() for _ in range(seq_length)]

        # Process one frame of one kinect camera
        def process_frame(cam_idx, frame_idx):
            rgb_frame = rgb_list[cam_idx][frame_idx]
            depth_frame = depth_list[cam_idx][frame_idx]
            return self.get_one_frame_pcd(cam_idx, rgb_frame, depth_frame)

        # Process one kinect
        def process_kinect(cam_idx):
            local_pcd_list = [None] * seq_length
            with ThreadPoolExecutor(max_workers=8) as frame_executor:
                # Add progress bar for frames processing
                with tqdm(total=seq_length, desc=f"Kinect {cam_idx} frames", position=cam_idx) as pbar:
                    results = []
                    for result in frame_executor.map(lambda idx: process_frame(cam_idx, idx), range(seq_length)):
                        results.append(result)
                        pbar.update(1)
                    for i, pcd in enumerate(results):
                        local_pcd_list[i] = pcd
            return local_pcd_list

        print("Processing Kinect cameras...")
        # Multi-threading to process each Kinect camera
        with ThreadPoolExecutor(max_workers=4) as kinect_executor:
            kinect_results = list(tqdm(kinect_executor.map(process_kinect, range(4)), 
                                total=4, desc="Kinect cameras"))

        print("Merging point clouds...")
        # Merge 4 kinect pcd with progress bar
        for cam_pcd_list in tqdm(kinect_results, desc="Merging Kinect data"):
            for i in range(seq_length):
                merged_pcd_list[i] += cam_pcd_list[i]

        self.merged_pcd_list = merged_pcd_list
        return

    def get_one_frame_pcd(self, cam_idx, rgb: torch.Tensor, depth: torch.Tensor) -> o3d.geometry.PointCloud:
        """
        undistort rgb & depth -> pcd -> pcd (in global)
        """
        undistorted_rgb = self.undistort_image(cam_idx, rgb.numpy())
        # undistorted_rgb = cv2.cvtColor(undistorted_rgb, cv2.COLOR_BGR2RGB)
        
        # Rectify depth distortion
        undistorted_depth = self.undistort_image(cam_idx, depth.numpy())

        pcd = create_point_cloud_from_rgb_and_depth(
            undistorted_rgb, undistorted_depth, 
            self.kinect_intrinsics_list[cam_idx]["matrix"], 
            self.kinect_intrinsics_list[cam_idx]["width"], 
            self.kinect_intrinsics_list[cam_idx]["height"]
        )

        # Transform pcd to global coordinate
        pcd.transform(self.kinect_extrinsics_list[cam_idx])

        if self.crop:
            pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(self.crop_bounding_box[0], self.crop_bounding_box[1]))
        
        return pcd

    def undistort_image(self, cam_idx, image: np.ndarray) -> np.ndarray:
        """
        Rectify the distortion of one iamge

        :param image: np.ndarray((3, 3))
        :return undistorted_image: the image that is rectifed distortion
        """
        camera_matrix = self.kinect_intrinsics_list[cam_idx]["matrix"]
        distortion_coeffs = self.kinect_intrinsics_list[cam_idx]["distortion"]

        h, w = image.shape[:2]

        # Rectify distortion
        mapx, mapy = cv2.initUndistortRectifyMap(
            camera_matrix, distortion_coeffs, None, None, (w, h), 5)
        undistorted_image = cv2.remap(
            image, mapx, mapy, interpolation=cv2.INTER_NEAREST)

        return undistorted_image

    def get_kinect_merged_pcd(self, frame_idx):
        """
        get the pcd of all kinects on the frame index

        :param frame_idx: the current frame index
        :return pcd: 
        """

        return self.merged_pcd_list[frame_idx]


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
    parser.add_argument("--obj_name", type=str,
                        help="the name of the object",
                        # required=True
    )
    parser.add_argument("--hand_arm_viz",
                        action="store_true",
                        help="hand_arm_viz",
                        default=False
    )
    parser.add_argument("--kinect_viz",
                        action="store_true",
                        help="kinect_viz",
                        default=False
    )    
    parser.add_argument("--realsense_viz",
                        action="store_true",
                        help="realsense_viz",
                        default=False
    )   
    parser.add_argument("--mano_viz",
                        action="store_true",
                        help="mano_viz",
                        default=False
    )  
    parser.add_argument("--object_viz",
                        action="store_true",
                        help="object_viz",
                        default=False
    )  
    parser.add_argument("--without_crop",
                        action="store_true",
                        help="viz pcd without crop",
                        default=False
    )
    parser.add_argument("--mask_pcd",
                        action="store_true",
                        help="viz masked pcd",
                        default=False
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


def main():
    args = init_and_get_args()
    # fix data_cache_folder_path everytime!!
    root_path = args.root_path    # "./data_cache"
    obj_name = args.obj_name

    without_crop = args.without_crop
    mask_pcd = args.mask_pcd

    if args.kinect_viz:
        kinect_data_handler = kinect_data(root_path, without_crop=without_crop)
    hand_arm_data_handler = hand_arm_data(root_path)

    if args.realsense_viz:
        realsense_data_handler = realsense_data(args.data_index, sync_config['realsense_begin_index_list'], sync_config['hand_arm_begin_index'], _data_cache_folder_path=data_cache_folder_path)

    mano_data_handler = mano_data(root_path)  

    # if args.object_viz:
    object_data_handler = object_data(root_path, obj_name)
    seq_length = object_data_handler.get_sequence_length()

    # Get the global timestamp list (reference realsense_0 & the first timestamp is the sync_frame_index)
    # global_time_stamp_list = realsense_data_handler.get_one_realsense_time_stamp_list(0)[sync_config['realsense_begin_index_list'][0]:]
    # global_time_stamp_list = kinect_data_handler.get_kinect_time_stamp_list(0)[sync_config['kinect_begin_index']:]

    #-------------------------------vis setting
    kinect_camera_view_parameter_path = "visualize_dataset/vis_view_params.json"
    kinect_camera_params = o3d.io.read_pinhole_camera_parameters(str(kinect_camera_view_parameter_path))
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    first_vis_flag = True

    # # 创建保存渲染结果的文件夹
    # output_dir = os.path.join(root_path, "rendered_frames_left_without_mesh")
    # os.makedirs(output_dir, exist_ok=True)


    # end_frame = len(global_time_stamp_list) if not args.end_frame else args.end_frame

    #-------------------------------viz
    # for index in np.arange(len(global_time_stamp_list)):
    # for index in range(args.start_frame, len(global_time_stamp_list)):
    for index in range(seq_length):
        print(colorama.Fore.GREEN + f"current frame_idx: {index}")

        if first_vis_flag:
            first_vis_flag = False
        else:
            vis.clear_geometries()

        if args.mano_viz:
            humanhand_mesh = mano_data_handler.get_mano_mesh(index)
            vis.add_geometry(humanhand_mesh)
            # if index == 0:
            #     o3d.io.write_triangle_mesh(f"visualize_dataset/hand_mesh.obj", humanhand_mesh)
        
        if args.object_viz:
            object_mesh = object_data_handler.get_object_mesh(index)
            vis.add_geometry(object_mesh)

        if args.hand_arm_viz:
            hand_arm_mesh = hand_arm_data_handler.get_hand_arm_mesh(index)
            vis.add_geometry(hand_arm_mesh)

        if args.realsense_viz:
            realsense_pcd_list = realsense_data_handler.get_camera_pcd_list(global_time_stamp_list[index])
            vis.add_geometry(realsense_pcd_list[0])
            vis.add_geometry(realsense_pcd_list[1])

        if args.kinect_viz:
            # pcd_list = kinect_data_handler.get_camera_pcd_list_with_time_stamp(global_time_stamp_list[index])
            # for pcd in pcd_list:
            #     vis.add_geometry(pcd)
            kinect_merged_pcd = kinect_data_handler.get_kinect_merged_pcd(index)
            vis.add_geometry(kinect_merged_pcd)
            # if index == 0:
            #     o3d.io.write_point_cloud(f"visualize_dataset/kinect_pcd.ply", kinect_merged_pcd)
                

        # -----------------------------
        # vis.get_render_option().point_size = 5.0  
        vis.get_view_control().convert_from_pinhole_camera_parameters(kinect_camera_params, allow_arbitrary=True)
        vis.poll_events()
        vis.update_renderer()

        # output_path = os.path.join(output_dir, f"{index:04d}.png")
        # vis.capture_screen_image(output_path, do_render=True)

        vis.run()


if __name__ == "__main__":
    main()
