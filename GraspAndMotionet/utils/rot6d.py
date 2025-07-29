import torch
import numpy as np
import transforms3d
import open3d as o3d
from scipy.spatial.transform import Rotation
import pytorch3d




def random_rot(device='cuda'):
    rot_angles = np.random.random(3) * np.pi * 2
    theta_x, theta_y, theta_z = rot_angles[0], rot_angles[1], rot_angles[2]
    Rx = torch.tensor([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]]).to(device)
    Ry = torch.tensor([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]]).to(device)
    Rz = torch.tensor([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]]).to(device)
    return (Rx @ Ry @ Rz).clone().detach()  # [3, 3]


def rot_to_orthod6d(rot):
    return rot.transpose(1, 2)[:, :2].reshape([-1, 6])


def get_rot6d_from_rot3d(rot3d):
    global_rotation = np.array(transforms3d.euler.euler2mat(rot3d[0], rot3d[1], rot3d[2]))
    return global_rotation.T.reshape(9)[:6]


def compute_rotation_matrix_from_ortho6d(poses):
    """
    Code from
    https://github.com/papagina/RotationContinuity
    On the Continuity of Rotation Representations in Neural Networks
    Zhou et al. CVPR19
    https://zhouyisjtu.github.io/project_rotation/rotation.html
    """
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3
        
    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3
        
    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix


def robust_compute_rotation_matrix_from_ortho6d(poses):
    """
    Instead of making 2nd vector orthogonal to first
    create a base that takes into account the two predicted
    directions equally
    """
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    y = normalize_vector(y_raw)  # batch*3
    middle = normalize_vector(x + y)
    orthmid = normalize_vector(x - y)
    x = normalize_vector(middle + orthmid)
    y = normalize_vector(middle - orthmid)
    # Their scalar product should be small !
    # assert torch.einsum("ij,ij->i", [x, y]).abs().max() < 0.00001
    z = normalize_vector(cross_product(x, y))

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    # Check for reflection in matrix ! If found, flip last vector TODO
    # assert (torch.stack([torch.det(mat) for mat in matrix ])< 0).sum() == 0
    return matrix


def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, v.new([1e-8]))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v/v_mag
    return v


def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        
    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)
        
    return out






def compute_rotation_matrix_analytical(source_point, target_point):
    source_point = source_point / np.linalg.norm(source_point)
    target_point = target_point / np.linalg.norm(target_point)
    
    # 计算旋转轴和角度
    v = np.cross(source_point, target_point)
    s = np.linalg.norm(v)
    c = np.dot(source_point, target_point)
    
    if s < 1e-10:  
        if c > 0:  
            return np.eye(3)
        else:  

            v = np.array([1, 0, 0]) if abs(source_point[0]) < 0.9 else np.array([0, 1, 0])
            v = v - np.dot(v, source_point) * source_point
            v = v / np.linalg.norm(v)
            
    v_x = np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])
    
    R = np.eye(3) + v_x + v_x @ v_x * ((1 - c) / (s * s))
    return R
def pcd_self_rotate(pcd, num = 12):
    points_array = np.asarray(pcd.cpu().numpy())
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(points_array)


    # 计算边界框
    obbox = o3d_pcd.get_oriented_bounding_box()
    obbx_rotation = np.asarray(obbox.R)
    # 获取边界框的主轴方向（假设柱状体的主轴是最长的那个）
    principal_axis_idx = np.argmax(obbox.extent)
    principal_axis = np.zeros((3,1))
    principal_axis[principal_axis_idx,0] = 1.0
    axis = np.dot(obbx_rotation, principal_axis).reshape(1,3)

    angles = np.radians(np.arange(0, 360, step = 360 / num)).reshape(num,1)


        
    rot_mat = torch.from_numpy(Rotation.from_rotvec(angles @ axis).as_matrix())
    return rot_mat


def uniform_rotation_matrics(global_rotate_num = 30,self_rotate_pcd = None,self_rotate_pcd_num = 0):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    pcd = mesh.sample_points_uniformly(number_of_points=10000)
    pcd = torch.from_numpy(np.asarray(pcd.points)).float()
    fps_points = pytorch3d.ops.sample_farthest_points(pcd.unsqueeze(0), K=global_rotate_num)[0][0]

    reference_point = fps_points[0] 
    other_points = fps_points[1:]   
    
    rotation_matrices = [torch.eye(3)]
    for point in other_points:
        R = compute_rotation_matrix_analytical(reference_point, point)
        R = torch.from_numpy(R)
        rotation_matrices.append(R)
    
    rotation_tensors = torch.stack(rotation_matrices)# [num, 3, 3]
    if self_rotate_pcd_num:
        self_rot_mat = pcd_self_rotate(self_rotate_pcd,self_rotate_pcd_num)
        rotation_tensors = rotation_tensors.unsqueeze(0).repeat(self_rot_mat.shape[0],1,1,1)
        self_rot_mat = self_rot_mat.unsqueeze(1).repeat(1,rotation_tensors.shape[1],1,1)
        rotation_tensors = (rotation_tensors @ self_rot_mat).reshape(-1,3,3)

    return rotation_tensors


    
