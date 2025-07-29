import json
import os
import pytorch_kinematics as pk
import torch.nn
import trimesh as tm
import urdf_parser_py.urdf as URDF_PARSER
from plotly import graph_objects as go
from pytorch_kinematics.urdf_parser_py.urdf import (URDF, Box, Cylinder, Mesh, Sphere)
import trimesh.sample
import open3d as o3d
from pytorch3d.transforms import axis_angle_to_matrix,Transform3d,rotation_6d_to_matrix
import numpy as np  
from pytorch3d.ops import knn_points
import torch.nn.functional as F
import pytorch3d
from typing import Dict
import transforms3d


class vis_mesh_type:
    trimesh = 0
    open3d = 1





class HandModel:
    def __init__(self, robot_name, urdf_filename, mesh_path,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                 hand_scale=2.,
                 baselink = None,
                 remove_wrist = False,
                 more_surface_points = True
                 ):
        '''
        all the rotation here are view as multiply to left 
        
        '''
        self.device = device
        self.robot_name = robot_name
        # prepare model
        self.robot = pk.build_chain_from_urdf(open(urdf_filename).read()).to(dtype=torch.float, device=self.device)
        if more_surface_points:
            # surface_pts_file_path = './assets/shadow_asserts/shadow_surface_pts/surface_pts.pth'
            surface_pts_file_path = "./assets/shadow_asserts/shadow_surface_pts/surface_pts.pth"
            every_link_sample_count = 128
        else:
            surface_pts_file_path = "./assets/shadow_asserts/shadow_surface_pts/min_surface_pts.pth"
            # surface_pts_file_path = './assets/shadow_asserts/shadow_surface_pts/min_surface_pts.pth'
            every_link_sample_count = 10
        

        if baselink is not None:
            root_frame = self.robot.find_frame(baselink) # will return the cpu item 
            self.robot = pk.chain.Chain(root_frame).to(dtype=torch.float, device=self.device)
            # you asign a root frame, but the frame is the child frame, the father frame is not account 
            # into it, but the joint is account
            self.palm_inv_transform = np.eye(4)
            self.palm_inv_transform[:3,3] = np.array([ 0.0000, -0.0100,  0.2470])
            self.palm_inv_transform = torch.from_numpy(np.linalg.inv(self.palm_inv_transform)).unsqueeze(0).to(torch.float32).to(self.device)
            self.remove_wrist = remove_wrist
            # when the baselink is the palm,you will use it

        self.robot_full = URDF_PARSER.URDF.from_xml_file(urdf_filename)
        # prepare contact point basis and surface point samples
        # self.no_contact_dict = json.load(open(os.path.join('data', 'urdf', 'intersection_%s.json'%robot_name)))
        # 'TODO:spenloss'
        self.keypoints = {
            # "rh_wrist": [],
            # "rh_palm": [],
            # "rh_ffknuckle": [],
            "rh_ffproximal": [[0, 0, 0.024]],
            "rh_ffmiddle": [[0, 0, 0], [0, 0, 0.025]],
            "rh_ffdistal": [[0, 0, 0.024]],
            # "rh_fftip": [],
            # "rh_mfknuckle": [],
            "rh_mfproximal": [[0, 0, 0.024]], 
            "rh_mfmiddle": [[0, 0, 0], [0, 0, 0.025]],
            "rh_mfdistal": [[0, 0, 0.024]],
            # "rh_mftip":[],
            # "rh_rfknuckle": [],
            "rh_rfproximal": [[0, 0, 0.024]], 
            "rh_rfmiddle": [[0, 0, 0], [0, 0, 0.025]],
            "rh_rfdistal": [[0, 0, 0.024]],
            # "rh_lfmetacarpal": [],
            # "rh_lfknuckle": [],
            "rh_lfproximal": [[0, 0, 0.024]],
            "rh_lfmiddle": [[0, 0, 0], [0, 0, 0.025]],
            "rh_lfdistal": [[0, 0, 0.024]],
            # "rh_lftip": [],
            # "rh_thbase": [], 
            "rh_thproximal": [[0, 0, 0.038]], 
            # "rh_thhub": [],
            "rh_thmiddle": [[0, 0, 0.032]], 
            "rh_thdistal": [[0, 0, 0.026]],
            # "rh_thtip":[]
        }
        
        self.base_dis_key_point = {
            "f_distal": [
                        [ 7.4092, -5.0441, 10.9599],
                        [-4.2443,  3.5357, 30.6710],
                        [-7.2797, -4.2617, 16.8451],
                        [ 7.1346, -0.8382, 24.3756],
                        [-3.0858, -7.3378,  7.0781],
                        [-1.7134, -5.1688, 25.8240],
                        [ 3.0772, -6.6101, 18.8368],
                        [ 3.1800,  0.0656, 31.6062],
                        [-7.2940, -0.1283, 24.0375],
                        [-2.8548, -7.2383, 13.2836]
                        ],
            "f_middle": [
                        [ 2.8051, -8.9449, 11.9081],
                        [-4.1032, -8.8298, 11.3706],
                        [-1.1935, -7.9284,  0.8216],
                        [-3.1775, -8.0140,  0.1409],
                        [ 3.6101, -8.8415, 13.3560],
                        [-0.2742, -6.7297, 24.1976],
                        [-3.5279, -8.7720, 14.2021],
                        [ 3.2427, -8.9725, 10.0511],
                        [ 6.7736, -7.0348,  1.2305],
                        [ 1.2472, -7.9404,  0.7266],
                        [-1.3920, -6.7526, 26.7140],
                        [-4.9288, -6.4934, 27.5103],
                        [ 6.0629, -7.0068,  2.5569],
                        [-0.2803, -4.9733, 29.6525],
                        [-3.9991, -7.8583, -1.3932],
                        [-3.6195, -5.4743, 29.0093],
                        [ 2.9652, -6.7086, 26.8799],
                        [-2.5953, -4.6634, 30.0503],
                        [-5.6127, -6.7362, 25.4604],
                        [ 3.6371, -7.9859,  0.3641]
            ],
            "f_proximal":[[ 4.2549, -8.9931, 24.0060],
                        [-0.1728, -9.9741, 28.8149],
                        [-0.4016, -9.9394, 27.8875],
                        [-6.6150, -7.4932, 24.5419],
                        [ 4.1861, -8.3105, 16.4116],
                        [ 6.2729, -7.7717, 27.9860],
                        [ 6.3269, -7.6847, 25.7188],
                        [ 6.2944, -7.7321, 22.5655],
                        [-0.4156, -8.4977, 16.4830],
                        [-4.3600, -8.3372, 17.0640]
                        ],
            "th_distal": [
                        [-7.7976,  2.0000, 31.1807],
                        [10.3636, -2.5945, 12.6054],
                        [-9.2751, -5.3443, 12.7352],
                        [ 8.9424,  3.2125, 28.9096],
                        [ 2.4543, -7.3976, 22.1617],
                        [ 1.3616, -2.9891, 32.9202],
                        [-6.9127, -5.1705, 24.4418],
                        [ 3.0332, -8.8284, 11.6626],
                        [ 9.6307, -2.0549, 21.4286],
                        [-4.0612, -8.1503, 17.4891]
                        ],
            "th_middle":[
                        [  -8.9967,   6.0022,   3.8358],
                        [  -8.0937,  -6.4849,   9.1191],
                        [  -8.4368,  -2.6608,  23.5721],
                        [  -8.6052,   4.1576,  15.5540],
                        [  -5.7045,   8.1706,  11.8752],
                        [  -9.8551,  -2.5015,  10.3311],
                        [  -2.1074, -10.3314,   7.5997],
                        [  -8.6495,  -2.8007,  20.3802],
                        [  -4.6096,   8.8808,  11.8244],
                        [  -6.3303,   7.8996,  23.1311]
                        ],
            # "th_proximal":[
            #             [  -9.2636,  -5.0265,  15.0830],
            #             [  -9.4676,   6.3865,   9.4502],
            #             [  -7.0439,  -7.0664,  28.3489],
            #             [ -12.6317,  -0.7874,   1.7740],
            #             [  -3.4622, -10.3062,  12.8241],
            #             [  -4.2088,   9.2904,  18.5275],
            #             [  -8.4013,   6.3664,  16.0117],
            #             [  -8.7836,   4.8914,  18.1536],
            #             [  -9.3265,   2.5616,  29.0031],
            #             [  -6.5950,  -7.5326,  23.5803]
            #             ],
                        
            "lfmetacarpal":[
                        [  4.9670, -11.0000,  41.1807],
                        [ -7.4695, -10.5640,  28.3394],
                        [ -5.4537, -10.9953,  22.0194],
                        [ -3.1178, -10.9973,  21.5453],
                        [ -3.0632, -10.9948,  49.1313],
                        [  0.1435, -10.9924,  39.0128],
                        [  0.0740, -10.9970,  26.6185],
                        [ -3.1690, -10.9971,  21.8211],
                        [ -4.0793, -10.9939,  26.7540],
                        [  5.2239, -11.0027,  53.7838]]
        }
        # let the dim adapt to the batch size, so it can use the function bmm
        self.dis_key_point = {}

        self.link_face_verts = {}
        # prepare geometries for visualization
        self.global_translation = None
        self.global_rotation = None
        self.softmax = torch.nn.Softmax(dim=-1)
        # prepare contact point basis and surface point samples
        self.surface_points = {}
        self.surface_points_normal = {}
        visual = URDF.from_xml_string(open(urdf_filename).read())
        self.mesh_verts = {}
        self.mesh_faces = {}

        self.canon_verts = []
        self.canon_faces = []
        self.idx_vert_faces = []
        self.face_normals = []

        if robot_name == 'shadowhand':
            self.palm_toward = torch.tensor([0., -1., 0., 0.], device=self.device).reshape(1, 1, 4)
        else:
            raise NotImplementedError
        skip_link_names = ["rh_forearm","rh_wrist"]

        for i_link, link in enumerate(visual.links):
            if baselink is not None and (link.name[:2] != "rh" or link.name in skip_link_names):  # use this way to let bi urdf adapt to this class
                continue
            # print(f"Processing link #{i_link}: {link.name}")
            # load mesh
            if len(link.visuals) == 0:
                continue
            if type(link.visuals[0].geometry) == Mesh:
                # print(link.visuals[0])
                filename = link.visuals[0].geometry.filename.split('/')
                filename[-1] = filename[-1].replace(".dae", ".obj")
                filename = filename[2:]
                filename = '/'.join(filename)
                mesh = tm.load(os.path.join(mesh_path, filename), force='mesh', process=False)
            elif type(link.visuals[0].geometry) == Cylinder:
                mesh = tm.primitives.Cylinder(
                    radius=link.visuals[0].geometry.radius, height=link.visuals[0].geometry.length)
            elif type(link.visuals[0].geometry) == Box:
                mesh = tm.primitives.Box(extents=link.visuals[0].geometry.size)
            elif type(link.visuals[0].geometry) == Sphere:
                mesh = tm.primitives.Sphere(
                    radius=link.visuals[0].geometry.radius)
            else:
                print(type(link.visuals[0].geometry))
                raise NotImplementedError
            try:
                scale = np.array(
                    link.visuals[0].geometry.scale).reshape([1, 3])
            except:
                scale = np.array([[1, 1, 1]])

            # Attention: multiply the original rot and tanslation
            try:
                rotation = transforms3d.euler.euler2mat(*link.visuals[0].origin.rpy)
                translation = np.reshape(link.visuals[0].origin.xyz, [1, 3])
                # print('---')
                # print(link.visuals[0].origin.rpy, rotation)
                # print('---')
            except AttributeError:
                rotation = transforms3d.euler.euler2mat(0, 0, 0)
                translation = np.array([[0, 0, 0]])
            
            if not os.path.exists(surface_pts_file_path):
                if link.name == "rh_palm":
                    if more_surface_points:
                        while True:# because this func would return the pts count not equal to the count you assign
                            pts, pts_face_index = trimesh.sample.sample_surface_even(mesh=mesh, count=150)
                            if np.array(pts).shape[0] == 128:
                                break   
                    else:
                        pts, pts_face_index = trimesh.sample.sample_surface_even(mesh=mesh, count=10)
                else:
                    while True:
                        pts, pts_face_index = trimesh.sample.sample_surface_even(mesh=mesh, count=every_link_sample_count)
                        if np.array(pts).shape[0] == every_link_sample_count:
                            break
                pts_normal = np.array([mesh.face_normals[x] for x in pts_face_index], dtype=float)
                pts *= scale
                pts = np.matmul(rotation, pts.T).T + translation
                # pts_normal = np.matmul(rotation, pts_normal.T).T
                pts = np.concatenate([pts, np.ones([len(pts), 1])], axis=-1)
                pts_normal = np.concatenate([pts_normal, np.ones([len(pts_normal), 1])], axis=-1)
                self.surface_points[link.name] = torch.from_numpy(pts).to(
                    device).float().unsqueeze(0)
                self.surface_points_normal[link.name] = torch.from_numpy(pts_normal).to(
                    device).float().unsqueeze(0)
                # print(self.dis_key_point)
            
            # if link.name in self.dis_key_point.keys():

                # temp_key_point = np.array(self.dis_key_point[link.name]) * scale
                # self.dis_key_point[link.name] = np.matmul(rotation , temp_key_point.T).T + translation 

            # visualization mesh
            self.mesh_verts[link.name] = np.array(mesh.vertices) * scale
            self.mesh_verts[link.name] = np.matmul(rotation, self.mesh_verts[link.name].T).T + translation
            self.mesh_faces[link.name] = torch.tensor(mesh.faces, dtype=torch.long).to(self.device)
            self.mesh_verts[link.name] = torch.from_numpy(self.mesh_verts[link.name]).to(torch.float).to(self.device)
            
            'TODO'# Use index_vertices_by_faces
            self.contact_pts_init(link.name,rotation,translation,scale)

        if os.path.exists(surface_pts_file_path):
            # print("use pre define surface points")
            surface_data = torch.load(surface_pts_file_path)
            # surface_normal_data = torch.load(surface_pts_normal_file_path)
            surface_data = self.dict_data_to_device(surface_data)
            # surface_normal_data = self.dict_data_to_device(surface_normal_data)
            self.surface_points = surface_data
            # self.surface_points_normal = surface_normal_data
            
        else:
            torch.save(self.surface_points,surface_pts_file_path)
            # torch.save(self.surface_points_normal,surface_pts_normal_file_path)
        for link_name in self.surface_points.keys():
            self.surface_points[link_name] = self.surface_points[link_name]
            # self.surface_points_normal[link_name] = self.surface_points_normal[link_name]

        # new 2.1
        self.revolute_joints = []
        for i in range(len(self.robot_full.joints)):
            if self.robot_full.joints[i].joint_type == 'revolute':
                self.revolute_joints.append(self.robot_full.joints[i])
        self.revolute_joints_q_mid = []
        self.revolute_joints_q_var = []
        self.revolute_joints_q_upper = []
        self.revolute_joints_q_lower = []
        for i in range(len(self.robot.get_joint_parameter_names())):
            for j in range(len(self.revolute_joints)):
                if self.revolute_joints[j].name == self.robot.get_joint_parameter_names()[i]:
                    joint = self.revolute_joints[j]
            assert joint.name == self.robot.get_joint_parameter_names()[i]
            self.revolute_joints_q_mid.append(
                (joint.limit.lower + joint.limit.upper) / 2)
            self.revolute_joints_q_var.append(
                ((joint.limit.upper - joint.limit.lower) / 2) ** 2)
            self.revolute_joints_q_lower.append(joint.limit.lower)
            self.revolute_joints_q_upper.append(joint.limit.upper)

        self.revolute_joints_q_lower = torch.Tensor(
            self.revolute_joints_q_lower).unsqueeze(1).to(device)
        self.revolute_joints_q_upper = torch.Tensor(
            self.revolute_joints_q_upper).unsqueeze(1).to(device)

        self.current_status = None

        self.scale = hand_scale


    def contact_pts_init(self,link_name,rotation,translation,scale):
        original_name = link_name
        self.mesh_verts[original_name] = torch.cat([self.mesh_verts[original_name],torch.ones(self.mesh_verts[original_name].shape[0],1,device = self.device)],dim = 1).unsqueeze(0)
        # spen 
        if original_name in self.keypoints.keys():
            self.keypoints[original_name] = torch.tensor(self.keypoints[original_name],device = self.device)
            self.keypoints[original_name] = torch.cat([self.keypoints[original_name],torch.ones(self.keypoints[original_name].shape[0],1,device = self.device)],dim = 1).unsqueeze(0)
        
        #  will skip the hub link

        link_name = link_name[3:]
        contact_name_key = ""
        fingers = ["ff","mf","rf","lf"]
        filter_links = ["knuckle","base","hub"]

        if link_name == "lfmetacarpal":
            contact_name_key = "lfmetacarpal"
        elif link_name == "thproximal":# th proximal don't need dis_keypoint
            return None
        else:
            if any([link in link_name for link in filter_links]):
                return None
            elif link_name[:2] == "th":
                contact_name_key = "th_"
            elif link_name[:2] in fingers:
                contact_name_key = "f_"
            else:# not the link that means
                return None
            contact_name_key += link_name[2:]

        target_contact_pts = np.array(self.base_dis_key_point[contact_name_key]) * scale
        target_contact_pts = np.matmul(rotation , target_contact_pts.T).T + translation 
        target_contact_pts = torch.tensor(target_contact_pts, device=self.device, dtype=torch.float32)
        
        target_contact_pts = torch.cat([target_contact_pts,torch.ones(target_contact_pts.shape[0],1,device = self.device)],dim = 1).unsqueeze(0)
        self.dis_key_point[original_name] = target_contact_pts



    def dict_data_to_device(self,dict_data):
        for key,value in dict_data.items():
            dict_data[key] = value.to(self.device)
        return dict_data
    
    def from_dict_to_qpose_tensor(self,qpose_dict):
        joint_order = self.robot.get_joint_parameter_names()
        qpose_tensor = torch.tensor([qpose_dict[joint_name] for joint_name in joint_order],device=self.device)
        return qpose_tensor

    def update_kinematics(self, q):
        # if the dim of the qpose is 27, then the qpose do not contains the rotation of the hand 
        # 27 = translation + qpose

        # 30 = translation(3) + rotation(3) + qpose(24)

        # my code designment is, the qpose input here must be [trans+24(full_joint_angle)] or [trans+rot+24(full_joint_angle)],no remove wrist joint_angle would be input
        
        self.global_translation = q[:, :3]
        if q.shape[1] == 27: # trans + wrist + angle
            self.global_rotation = axis_angle_to_matrix(torch.tensor([0., 0., 0.], device=self.device).repeat(q.shape[0], 1))
            self.current_status = self.robot.forward_kinematics(q[:, 3:])
        
        elif q.shape[1] == 25:# trans + angle_no_wrist

            self.global_rotation = axis_angle_to_matrix(torch.tensor([0., 0., 0.], device=self.device).repeat(q.shape[0], 1))
            hand_full_angle = torch.cat([torch.zeros(q.shape[0],2).to(q.device),q[:, 3:]],dim=1)
            self.current_status = self.robot.forward_kinematics(hand_full_angle)

        elif q.shape[1] == 33:# trans + 6d rot, + 24 qpose

            self.global_rotation = rotation_6d_to_matrix(q[:, 3:9])
            self.current_status = self.robot.forward_kinematics(q[:, 9:])

        elif q.shape[1] == 30: # trans + axis_angle rot + wrist + angle 30

            self.global_rotation = axis_angle_to_matrix(q[:, 3:6])# [...,3,3]
            self.current_status = self.robot.forward_kinematics(q[:, 6:])
        else:
            raise NotImplementedError("the hand pose update type is not implemented")
        if self.remove_wrist:
            for key,value in self.current_status.items():
                self.current_status[key] = Transform3d(matrix = torch.matmul(self.palm_inv_transform , value.get_matrix()))
    
    def update_motion_kinematics(self,q):# remove wrist
        self.global_translation = q[:, :3]
        self.global_rotation = rotation_6d_to_matrix(q[:, 3:9])
        self.current_status = self.robot.forward_kinematics(q[:, 9:])
        for key,value in self.current_status.items():
            self.current_status[key] = Transform3d(matrix = torch.matmul(self.palm_inv_transform , value.get_matrix()))
    
    #"TODO:penloss_sdf"
    def pen_loss_sdf(self,obj_pcd: torch.Tensor,q=None ,test = False):
        from csdf import index_vertices_by_faces, compute_sdf
        if len(self.link_face_verts) == 0:
            for link_name in self.mesh_verts.keys():
                self.link_face_verts[link_name] = index_vertices_by_faces(self.mesh_verts[link_name], self.mesh_faces[link_name]).to(self.device).float()
                
        penetration = []
        if q is not None:
            self.update_kinematics(q)
        obj_pcd = obj_pcd.float()
        global_translation = self.global_translation.float()
        global_rotation = self.global_rotation.float()
        obj_pcd = (obj_pcd - global_translation.unsqueeze(1)) @ global_rotation
        # self.save_point_cloud(obj_pcd[1].detach().cpu().numpy(), f"{1}_point_cloud.ply")
        for link_name in self.link_face_verts:
            trans_matrix = self.current_status[link_name].get_matrix()
            obj_pcd_local = (obj_pcd - trans_matrix[:, :3, 3].unsqueeze(1)) @ trans_matrix[:, :3, :3]
            obj_pcd_local = obj_pcd_local.reshape(-1, 3)
            hand_face_verts = self.link_face_verts[link_name].detach()
            dis_local, _, dis_signs, _, _ = compute_sdf(obj_pcd_local, hand_face_verts)
            dis_local = torch.sqrt(dis_local + 1e-8)#eval
            penloss_sdf = dis_local * (-dis_signs)
            penetration.append(penloss_sdf.reshape(obj_pcd.shape[0], obj_pcd.shape[1]))  # (batch_size, num_samples)
            # self.save_point_cloud(obj_pcd_local.reshape(obj_pcd.shape[0], -1,3)[1].detach().cpu().numpy(), f"{link_name}_point_cloud.ply")
            # self.save_mesh(hand_face_verts.detach().cpu().numpy(), f"{link_name}_mesh.ply")
        # penetration = torch.max(torch.stack(penetration), dim=0)[0]
        # loss_pen_sdf = penetration[penetration > 0].sum() / obj_pcd.shape[0]
        if test:
            distances = torch.max(torch.stack(penetration, dim=0), dim=0)[0]
            return max(distances.max().item(), 0)
        
        penetration = torch.stack(penetration)
        # penetration = penetration.max(dim=0)[0]
        loss = penetration[penetration > 0].sum() / (penetration.shape[0]* penetration.shape[1])# distances[distances > 0].sum() / batch_size
        # print('eval:' ,max(penetration.max().item(), 0)) ###eval
        # print('penetration_sdf: ', penetration)
        return loss
    
    #"TODO:spenloss"
    def get_keypoints(self, q=None, downsample=True):
        return self.transform_dict_items(self.keypoints,q)

    def get_dis_keypoints(self, q=None, downsample=True):
        # TODO 将dis keypoint中的点转成 torch tensor 和转移到gpu 设备上的工作转移到 init 中完成
        return self.transform_dict_items(self.dis_key_point,q)

    def transform_dict_items(self,dict_items,q = None):
        # the item in dict_items are Homogeneous
        if q is not None:
            self.update_kinematics(q)
        points = []
        for link_name in dict_items.keys() :
            trans_matrix = self.current_status[link_name].get_matrix()
            points.append(dict_items[link_name] @ trans_matrix.transpose(1, 2))
        points = torch.cat(points, 1)
        points = points[...,:3] @ self.global_rotation.float().transpose(1, 2) + self.global_translation.unsqueeze(1)#  add points num size
        return points * self.scale

    def transform_dict_items_without_q(self,dict_items):
        # the item in dict_items are Homogeneous

        points = []
        for link_name in dict_items.keys() :
            trans_matrix = self.current_status[link_name].get_matrix()
            points.append(dict_items[link_name] @ trans_matrix.transpose(1, 2))
        points = torch.cat(points, 1)
        points = points[...,:3] @ self.global_rotation.float().transpose(1, 2) + self.global_translation.unsqueeze(1)#  add points num size
        return points * self.scale


    def get_surface_points(self, q=None):
        '''
        the point in the surface is sampled on every link with even sampling 128 points,
        so, the link with bigger volume, the points in this link will be more sparse,
        so, the points on the fingers will be very dense, and on the wrist will be very sparse
        '''
        return self.transform_dict_items(self.surface_points,q)

    def get_surface_points_without_q(self):
        '''
        the point in the surface is sampled on every link with even sampling 128 points,
        so, the link with bigger volume, the points in this link will be more sparse,
        so, the points on the fingers will be very dense, and on the wrist will be very sparse
        '''
        return self.transform_dict_items_without_q(self.surface_points)


    
    def get_palm_points(self, q=None):
        palm_pcd_dict = {}
        palm_pcd_dict['palm'] = self.surface_points['palm']
        return self.transform_dict_items(palm_pcd_dict)


    def get_palm_toward_point(self, q=None):
        if q is not None:
            self.update_kinematics(q)
        link_name = 'palm'
        trans_matrix = self.current_status[link_name].get_matrix()
        palm_toward_point = torch.matmul(trans_matrix, self.palm_toward.transpose(1, 2)).transpose(1, 2)[..., :3]
        palm_toward_point = torch.matmul(self.global_rotation, palm_toward_point.transpose(1, 2)).transpose(1, 2)

        return palm_toward_point.squeeze(1)

    def get_palm_center_and_toward(self, q=None):
        if q is not None:
            self.update_kinematics(q)

        palm_surface_points = self.get_palm_points()
        palm_toward_point = self.get_palm_toward_point()

        palm_center_point = torch.mean(palm_surface_points, dim=1, keepdim=False)
        return palm_center_point, palm_toward_point

    def get_surface_points_and_normals(self, q=None):
        '''
        because the translation won't change the normal of the surface points, so we can just use the global rotation to rotate the normal
        '''
        if q is not None:
            self.update_kinematics(q=q)
        surface_points = self.transform_dict_items(self.surface_points)
        # surface_normals = self.transform_dict_items(self.surface_points_normal)
        return surface_points#,surface_normals


    def get_meshes_from_q(self, q=None, type = vis_mesh_type.trimesh,batch_mode = False):
        # the wrist and forearm is not included in the mesh_verts
        '''
        all the qpose get in here must be the torch
        defoult is removed the wrist ,
        
        this method only output one mesh at once
        '''
        data = []
        link_points_list = []
        link_face_list = []
        if q is not None: self.update_kinematics(q)     
        for link_name in self.mesh_verts:
            trans_matrix = self.current_status[link_name].get_matrix()
            transformed_v = self.mesh_verts[link_name]
            
            transformed_v =  (transformed_v @ trans_matrix.transpose(1,2))[...,:3]
            transformed_v = transformed_v @ self.global_rotation.float().transpose(1, 2) + self.global_translation.unsqueeze(1)

            transformed_v = transformed_v * self.scale
            link_points_list.append(transformed_v)
            link_face_list.append(self.mesh_faces[link_name])

        if batch_mode:
            return link_points_list,link_face_list
        
        #output one hand mesh
        for link_idx in np.arange(len(link_face_list)):
            pcd = link_points_list[link_idx].squeeze(0).cpu()
            face = link_face_list[link_idx].cpu()
            if type == vis_mesh_type.trimesh:
                data.append(tm.Trimesh(vertices=pcd, faces=face))
            else:# type = open3d
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(pcd)
                mesh.triangles = o3d.utility.Vector3iVector(face)
                data.append(mesh)

        complete_mesh = None
        if type == vis_mesh_type.open3d:
            complete_mesh = o3d.geometry.TriangleMesh()
            for mesh in data:
                complete_mesh += mesh
            complete_mesh.compute_vertex_normals()
        else:# trimesh
            complete_mesh = tm.util.concatenate(data)
        
        return complete_mesh

    def get_plotly_data(self, q=None, i=0, color='lightblue', opacity=1.):
        data = []
        if q is not None: self.update_kinematics(q)
        for idx, link_name in enumerate(self.mesh_verts):
            trans_matrix = self.current_status[link_name].get_matrix()
            trans_matrix = trans_matrix[min(len(trans_matrix) - 1, i)].detach().cpu().numpy()
            v = self.mesh_verts[link_name]
            transformed_v = np.concatenate([v, np.ones([len(v), 1])], axis=-1)
            transformed_v = np.matmul(trans_matrix, transformed_v.T).T[..., :3]
            transformed_v = np.matmul(self.global_rotation[i].detach().cpu().numpy(),
                                      transformed_v.T).T + np.expand_dims(
                self.global_translation[i].detach().cpu().numpy(), 0)
            transformed_v = transformed_v * self.scale
            f = self.mesh_faces[link_name]
            data.append(
                go.Mesh3d(x=transformed_v[:, 0], y=transformed_v[:, 1], z=transformed_v[:, 2], i=f[:, 0], j=f[:, 1],
                          k=f[:, 2], color=color, opacity=opacity))
        return data




def pen_loss(obj_pcd: torch.Tensor, obj_pcd_normal: torch.Tensor, hand_pcd: torch.Tensor):
    """
    Calculate the penalty loss based on point cloud and normal.
    calculate the mean max penetration loss
    :param obj_pcd_nor: B x N_obj x 6 (object point cloud with normals)
    :param hand_pcd: B x N_hand x 3 (hand point cloud)
    :return: pen_loss (scalar)
    """

    # Compute K-nearest neighbors
    knn_result = knn_points(hand_pcd, obj_pcd, K=1, return_nn=True)
    distances = knn_result.dists
    indices = knn_result.idx

    distances = distances.sqrt()
    # Extract the closest object points and normals
    hand_obj_points = torch.gather(obj_pcd, 1, indices.expand(-1, -1, 3))
    hand_obj_normals = torch.gather(obj_pcd_normal, 1, indices.expand(-1, -1, 3))
    # Compute the signs
    hand_obj_signs = ((hand_obj_points - hand_pcd) * hand_obj_normals).sum(dim=2)
    hand_obj_signs = (hand_obj_signs > 0.).float()
    # Compute collision value
    collision_value = (hand_obj_signs * distances.squeeze(2)).max(dim=1).values
    # max_col_index = torch.argmax(collision_value)
    # print(collision_value[max_col_index])
    pen_value = collision_value.mean()
    return pen_value #, max_col_index



# dis_loss
def dis_loss(dis_points, obj_pcd: torch.Tensor, thres_dis = 0.02 ):
    '''
    cd loss
    if the hand is around the object near 2cm, then the hand will be attracted to the surface of the object
    '''
    dis_points = dis_points.to(dtype=torch.float32)
    obj_pcd = obj_pcd.to(dtype=torch.float32)
    dis_pred = pytorch3d.ops.knn_points(dis_points, obj_pcd).dists[:, :, 0] # 64*140  # squared chamfer distance from object_pc to contact_candidates_pred
    small_dis_pred = dis_pred < thres_dis ** 2# 64*140
    # dis_loss = dis_pred[small_dis_pred].sqrt().sum() / dis_points.shape[0]
    # dis_loss = torch.sigmoid(-1*(dis_pred[small_dis_pred].sqrt()/dis_pred[small_dis_pred].sqrt()).sum()/(dis_points.shape[0]*dis_points.shape[1]))- 0.2689
    # dis_loss = dis_pred.sqrt().max(dim=1)[0].mean()  # / (small_dis_pred.sum() + 1e-4)
    dis_loss = dis_pred[small_dis_pred].sqrt().sum() / (small_dis_pred.sum().item() + 1e-5)#1
    return dis_loss



# all the global transform is represented as axis_angle
class hand_loss:
    def __init__(self,
                 _device,
                 loss_type = "l2",
                 dis_loss_weight = 1000,
                 pen_loss_weight = 100,
                 hand_pose_loss_weight = 10000,
                 hand_surface_point_mse_loss_weight = 10000):
        
        self.device = _device
        self.hand_model = get_e3m5_handmodel(device = _device,more_surface_points= True)

        if loss_type == 'l1':
            self.criterion = F.l1_loss
        elif loss_type == 'l2':
            self.criterion = F.mse_loss
        else:
            raise Exception('Unsupported loss type.')

        self.dis_loss_weight = dis_loss_weight
        self.pen_loss_weight = pen_loss_weight
        self.hand_pose_loss_weight = hand_pose_loss_weight
        self.hand_surface_point_mse_loss_weight = hand_surface_point_mse_loss_weight


    def cvae_loss(self, batch_size, mean, log_var):
        '''
        :param recon_x: reconstructed hand xyz [B,778,3]
        :param x: ground truth hand xyz [B,778,3] #[B,778,6]
        :param mean: [B,z]
        :param log_var: [B,z]
        calculate cd loss 
        :return:
        '''
        # KLD loss
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / batch_size * 10.0
        return KLD
        

    def cal_loss(self,
                 data_dict: Dict,
                 pred_x0,
                 cvae_loss_data: Dict = None,
                 ):


        loss = 0

        gt_hand_qpose = data_dict["original_hand_qpos"].to(torch.float32)
        batch_size = gt_hand_qpose.shape[0]

        obj_pcd = data_dict['obj_pcd'].to(self.device)


        # only this loss don't need to concat the rotation
        hand_qpose_loss =self.criterion(gt_hand_qpose, pred_x0).sum() / batch_size * self.hand_pose_loss_weight
        loss += hand_qpose_loss
        print(f'hand_qpose_loss:{hand_qpose_loss}')

        gt_hand_pcd = self.hand_model.get_surface_points(q=gt_hand_qpose).to(dtype=torch.float32)
        perd_hand_pcd = self.hand_model.get_surface_points(q=pred_x0).to(dtype=torch.float32)
        # gt_hand_pcd = data_dict["hand_surface_points"]        

        obj_pcd_normal = data_dict['obj_pcd_normal']
        pen_loss_value = pen_loss(obj_pcd, obj_pcd_normal, perd_hand_pcd)* self.pen_loss_weight
        loss += pen_loss_value
        print(f'pen_loss_value:{pen_loss_value}')

        dis_keypoint = self.hand_model.get_dis_keypoints(q=pred_x0)
        dis_loss_value = dis_loss(dis_keypoint, obj_pcd)* self.dis_loss_weight
        loss += dis_loss_value
        print(f'dis_loss_value:{dis_loss_value}')


        hand_surface_point_mse_loss_value = self.criterion(gt_hand_pcd, perd_hand_pcd)
        hand_surface_point_mse_loss_value = hand_surface_point_mse_loss_value * self.hand_surface_point_mse_loss_weight
        loss += hand_surface_point_mse_loss_value
        print(f"hand_surface_point_mse_loss_value {hand_surface_point_mse_loss_value}")



        cvae_loss = self.cvae_loss(
                        batch_size,
                        cvae_loss_data["mean"],
                        cvae_loss_data["log_var"]) # self,recon_x, x, mean, log_var, mode='train'):
        loss += cvae_loss 
        print(f"cvae_loss:{cvae_loss}")
        
        print(f"loss sum:{loss}")
        return {
            'loss': loss,
            "hand_qpose_loss":hand_qpose_loss,
            "dis_loss_value":dis_loss_value,
            "pen_loss_value":pen_loss_value,
            "cvae_loss":cvae_loss,
            "hand_surface_point_mse_loss_value":hand_surface_point_mse_loss_value}




def get_e3m5_handmodel( device = "cpu", hand_scale=1., robot='shadowhand',remove_wrist = True, more_surface_points = False):
    # urdf_assets_meta = json.load(open("/home/lab4dv/youzhuo/DynamicGraspTrainingCode-main/assets/shadow_asserts/bi_shadow_hand_config/e3m5_urdf_assets_meta.json"))
    urdf_assets_meta = json.load(open("assets/shadow_asserts/bi_shadow_hand_config/e3m5_urdf_assets_meta_remote.json"))
    urdf_path = urdf_assets_meta['urdf_path'][robot]
    meshes_path = urdf_assets_meta['meshes_path'][robot]
    hand_model = HandModel(robot, urdf_path, meshes_path, 
                           device=device, 
                           hand_scale=hand_scale,
                        #    baselink="rh_wrist",
                           baselink="rh_wrist",
                           remove_wrist = remove_wrist,
                           more_surface_points = more_surface_points)# let the surface point on the hand is not too dense,so can input to the motion net
    return hand_model
