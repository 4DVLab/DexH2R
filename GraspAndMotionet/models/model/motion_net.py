import sys, os
import numpy as np
import torch
from models.model.pointnet_encoder import PointNetEncoder
from torch import nn
from omegaconf import DictConfig
from pytorch3d.transforms import rotation_6d_to_matrix,matrix_to_rotation_6d
from utils.e3m5_hand_model import get_e3m5_handmodel
from models.base import MODEL
from os.path import join as pjoin
import trimesh as tm
from pprint import pprint   




def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def compute_qpose_rotation_diff(qpose,target_qpose_pool):
    '''
    param:
        qpose:              [seq_len,33] translation,rotation_6d
        target_qpose_pool   [seq_len,pool_num,33]
    return 
        qpose_rotation_diff [seq_len,pool_num]
    '''
    
    qpose_rotation = rotation_6d_to_matrix(qpose[...,3:9]).unsqueeze(1)
    target_qpose_pool_rotation = rotation_6d_to_matrix(target_qpose_pool[...,3:9])
    R_diff = qpose_rotation @ target_qpose_pool_rotation.transpose(-2, -1)
    R_diff_trace = torch.diagonal(R_diff, dim1=-2, dim2=-1).sum(-1)

    rot_dist = torch.acos(torch.clamp((R_diff_trace - 1) / 2, -1, 1)) # [seq_len,pool_num]
    return rot_dist



def find_nearest_final_qpose_using_fourty_trans_top_rotation(qpose,target_qpose_pool):
    '''
    param:
        qpose:              [seq_len,30]
        target_qpose_pool   [seq_len,30,30]
    return 
        selected_final_qpose [seq_len,30]
    '''

    seq_len = qpose.shape[0]

    batch_qpose_rot_diff = compute_qpose_rotation_diff(qpose,target_qpose_pool)# [seq_len,pool_num]

    # only use trans
    trans_diff = (qpose.unsqueeze(1)[...,:3] - target_qpose_pool[...,:3]).norm(dim=-1)# [seq_len,pool_num]

    top_k = 40
    _, top_k_indices = torch.topk(trans_diff, top_k, dim=1, largest=False)  # [seq_len,5]
    batch_indices = torch.arange(seq_len).unsqueeze(1).expand(-1, top_k)  # [seq_len,5]
    filtered_trans_diff = batch_qpose_rot_diff[batch_indices, top_k_indices]  # [seq_len,5]

    _, min_local_indices = torch.min(filtered_trans_diff, dim=1)  # [seq_len]


    final_indices = top_k_indices[torch.arange(seq_len), min_local_indices]  # [seq_len]
    selected_final_qpose = target_qpose_pool[torch.arange(seq_len), final_indices, :]
    selected_final_qpose_rot_diff = batch_qpose_rot_diff[torch.arange(seq_len), final_indices]
    selected_final_qpose_trans_diff = trans_diff[torch.arange(seq_len), final_indices]
    #,final_indices
    return selected_final_qpose,selected_final_qpose_rot_diff,selected_final_qpose_trans_diff






def qpose_align_transform(qpose_tensor,inv_center_hand_rot_mat,inv_center_hand_trans):
    '''
    parame:
        qpose_tensor:[batchsize,wind,qpose_dim]
        inv_center_hand_rot_mat: [batchsize,3,3]
        inv_center_hand_trans: [batchsize,3]
    return:
        qpose_align_tensor: [batchsize,wind,qpose_dim]
    '''
    inv_center_hand_rot_mat = inv_center_hand_rot_mat.unsqueeze(1) # [batchsize,1,3,3]
    inv_center_hand_trans = inv_center_hand_trans.unsqueeze(-1).unsqueeze(1) # [batchsize,1,3,1]
    batch_size,wind_size = qpose_tensor.shape[:2]
    qpose_tensor_rot = matrix_to_rotation_6d(inv_center_hand_rot_mat @ rotation_6d_to_matrix(qpose_tensor[...,3:9].reshape(-1,6)).view(batch_size,wind_size,3,3))#[batchsize,wind,6]
    qpose_tensor_trans = (inv_center_hand_rot_mat @ qpose_tensor[...,:3].unsqueeze(-1) - inv_center_hand_trans).squeeze(-1)#[batchsize,wind,3]
    qpose_align_tensor = torch.cat([qpose_tensor_trans,qpose_tensor_rot,qpose_tensor[...,9:]],dim = -1)
    return qpose_align_tensor


def batch_wind_data_align_to_wind_center(batch_wind_hand_qpose,batch_final_qpose,batch_obj_pcd,current_frame_index):
    '''
    param:
        batch_wind_hand_qpose [batch,wind_len,qpose_dim]
        batch_final_qpose [batch,qpose_dim]
        batch_obj_pcd [batch,wind_len, pcd_num,3] 
    return:

    '''

    center_hand_rot_mat = rotation_6d_to_matrix(batch_wind_hand_qpose[:,current_frame_index,3:9])# [batchsize,3,3]
    center_hand_trans = batch_wind_hand_qpose[:,current_frame_index,:3].unsqueeze(-1)# [batchsize,3,1]

    inv_center_hand_rot_mat = center_hand_rot_mat.permute((0,2,1)) # [batchsize,3,3]
    inv_center_hand_trans = (inv_center_hand_rot_mat @ center_hand_trans).squeeze(-1) # [batchsize,3]

    batch_obj_pcd = pcd_align_transform(batch_obj_pcd,center_hand_rot_mat,inv_center_hand_trans) 

    batch_wind_hand_qpose = qpose_align_transform(batch_wind_hand_qpose,inv_center_hand_rot_mat,inv_center_hand_trans)
    batch_final_qpose = qpose_align_transform(batch_final_qpose.unsqueeze(1),inv_center_hand_rot_mat,inv_center_hand_trans).squeeze(1)# remove the wind dim

    return batch_wind_hand_qpose,batch_final_qpose,batch_obj_pcd

    
def pcd_align_transform(pcd,center_hand_rot_mat,inv_center_hand_trans):
    '''
    param:
        pcd: [batchsize,wind_len,pcd_num,3]
        center_hand_rot_mat: [batchsize,3,3]
        inv_center_hand_trans: [batchsize,3]
    return:
        pcd: [batchsize,wind_len,pcd_num,3]
    '''
    batch_size,wind_len,pcd_num,_ = pcd.shape

    pcd = (pcd.view(batch_size,wind_len,pcd_num,1,3) @ center_hand_rot_mat.view(batch_size,1,1,3,3) - inv_center_hand_trans.view(batch_size,1,1,1,3)).squeeze(-2)
    return pcd


class ResBlock(nn.Module):

    def __init__(self,
                 Fin,
                 Fout,
                 n_neurons=256):

        super(ResBlock, self).__init__()
        self.Fin = Fin
        self.Fout = Fout

        self.fc1 = nn.Linear(Fin, n_neurons)
        self.bn1 = nn.BatchNorm1d(n_neurons)

        self.fc2 = nn.Linear(n_neurons, Fout)
        self.bn2 = nn.BatchNorm1d(Fout)

        if Fin != Fout:
            self.fc3 = nn.Linear(Fin, Fout)

        self.ll = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, final_nl=True):
        Xin = x if self.Fin == self.Fout else self.ll(self.fc3(x))

        Xout = self.fc1(x)  # n_neurons
        Xout = self.bn1(Xout)
        Xout = self.ll(Xout)
        Xout = self.fc2(Xout)
        Xout = self.bn2(Xout)
        Xout = Xin + Xout
        if final_nl:
            return self.ll(Xout)
        return Xout



class motion_net_block(nn.Module):
    def __init__(self,input_dim,future_frames,use_dropout):
        super(motion_net_block, self).__init__()
        n_neurons=2048
        self.future_frames = future_frames
        # self.hand_surface_point_num = hand_surface_point_num

        self.dec_bn1 = nn.BatchNorm1d(input_dim)  
        self.dec_rb1 = ResBlock(input_dim, n_neurons)
        self.dec_rb2 = ResBlock(n_neurons + input_dim, n_neurons//2)
        self.dec_rb3 = ResBlock(n_neurons//2, n_neurons//2)
        self.dec_rb4 = ResBlock(n_neurons//2, n_neurons)
        self.dec_pose = nn.Linear(n_neurons, 28 * self.future_frames)# it don't have the wrist 
        # self.dec_xyz = nn.Linear(n_neurons, self.hand_surface_point_num*3* self.future_frames)
        self.dec_trans = nn.Linear(n_neurons, 3*self.future_frames)
        self.qpose_remove_wrist_dim = 22
        # self.dec_dist = nn.Linear(n_neurons, self.hand_surface_point_num*3* self.future_frames)
        self.dout = nn.Dropout(p=0.3, inplace=False) if use_dropout else nn.Identity()

    def forward(self,data):

        batchsize = data.shape[0]

        X0 = self.dec_bn1(data)
        X  = self.dec_rb1(X0, True)
        X  = self.dout(X)
        X  = self.dec_rb2(torch.cat([X0, X], dim=1), True)

        X = self.dout(X)
        X  = self.dec_rb3(X)
        X = self.dout(X)
        X = self.dec_rb4(X)


        rot_joint_angle_pose = self.dec_pose(X).view(batchsize,self.future_frames,28)# batch future (qpose_dim + 6)
        pose_add_wrist = torch.cat([rot_joint_angle_pose[..., :6], # rotation dim is 6d 
                                    torch.zeros((rot_joint_angle_pose.shape[0],rot_joint_angle_pose.shape[1],2)).to(rot_joint_angle_pose.device), 
                                    rot_joint_angle_pose[..., 6:]], dim=-1)
        trans = self.dec_trans(X).view(batchsize,self.future_frames,3)
        hand_full_qpose = torch.cat([trans,pose_add_wrist],dim = -1)
        # xyz = self.dec_xyz(X).view(batchsize,self.future_frames,self.hand_surface_point_num,3)
        # rh2last = self.dec_dist(X).view(batchsize,self.future_frames,self.hand_surface_point_num,3)

        return hand_full_qpose #xyz#, rh2last



# no matter how manny points input the point net, the output always be 1024
@MODEL.register()
class motion_net(nn.Module):
    def __init__(self,
                 cfg: DictConfig,
                 *args,
                 **kwargs):
        super().__init__()

        self.hand_model = get_e3m5_handmodel(get_device(),more_surface_points=False)

        
        self.qpose_dim = cfg.qpose_dim
        self.past_frames = cfg.past_frames # 24
        self.future_frames = cfg.future_frames
        # self.hand_surface_point_num = cfg.hand_surface_point_num

        self.LossL2 = nn.MSELoss(reduction='mean')
        self.LossL1 = nn.L1Loss(reduction='mean')
        self.pcd_num = cfg.pcd_num  # the sample_points is actually the points num of the pcd 
        self.window_size = self.past_frames + self.future_frames
        self.obj_pcd_encode = PointNetEncoder( channel=3)
        self.motion_net_block = motion_net_block(input_dim = cfg.input_dim,
                                                 future_frames = self.future_frames,
                                                 use_dropout = cfg.use_dropout
                                                 )
        self.align_to_current_frame = cfg.align_to_current_frame
        self.seq_index = 0

        self.use_qpose_hand_pcd_loss = cfg.use_qpose_hand_pcd_loss
        self.use_qpose_loss = cfg.use_qpose_loss

    def sample(self, data):
        '''
        target pose = final pose
        '''
        
        batchsize = data["wind_obj_pcd"].shape[0]

        history_slice = slice(self.past_frames)
        current_index = self.past_frames - 1

        obj_pcd_history = data["wind_obj_pcd"][:,history_slice]
        hand_qpose_history = data["wind_hand_qpose"][:,history_slice]

        current_hand_qpose = hand_qpose_history[:,current_index]
        current_target_qpose_pool = data["final_qpose_pool"][:,current_index]
        batch_current_target_qpose,_,_ = find_nearest_final_qpose_using_fourty_trans_top_rotation(current_hand_qpose, current_target_qpose_pool)

        
        if self.align_to_current_frame:
            hand_qpose_history,batch_current_target_qpose,obj_pcd_history = batch_wind_data_align_to_wind_center(hand_qpose_history,batch_current_target_qpose,obj_pcd_history,current_index)

        hand_surface_pcd_history = self.hand_model.get_surface_points(hand_qpose_history.view(-1,self.qpose_dim)).view(batchsize,self.past_frames,-1,3)
        batch_current_hand_surface_points = hand_surface_pcd_history[:,current_index]
        current_hand_velocoty = batch_current_hand_surface_points - hand_surface_pcd_history[:,current_index - 1]
        obj_embed_history = self.obj_pcd_encode(obj_pcd_history.view((batchsize * self.past_frames,self.pcd_num,3))).view(batchsize,self.past_frames * 1024)# [batchsize,5* 1024]
        hand_indices = torch.cat([torch.arange(0, 9), torch.arange(11, 33)])# remove the hand wrist
        hand_qpose_remove_wrist_history = hand_qpose_history[..., hand_indices]
        batch_final_qpose_surface_points = self.hand_model.get_surface_points(batch_current_target_qpose)

        batch_current_to_final_hand_points_offset = batch_current_hand_surface_points - batch_final_qpose_surface_points # [batchsize,pcd_num,3]

        view_shape = [batchsize,-1]
        input_data = torch.cat([
                           hand_qpose_remove_wrist_history.view(*view_shape),
                           obj_embed_history.view(*view_shape),
                           current_hand_velocoty.view(*view_shape),
                           batch_current_hand_surface_points.view(*view_shape),
                           batch_current_to_final_hand_points_offset.view(*view_shape)
                           ], dim=1)

        delta_hand_full_qpose = self.motion_net_block(input_data)# hand full qpose add the wrist''', delta_rh2last         delta_hand_surface_pcd_xyz'''
        '''
        delta_hand_full_qpose: [batchsize,future_frames,qpose_dim]
        delta_hand_surface_pcd_xyz: [batchsize,future_frames,pcd_num,3]
        delta_rh2last: [batchsize,future_frames,pcd_num,3]
        '''
        
        delta_hand_rot = rotation_6d_to_matrix(delta_hand_full_qpose[...,3:9].view(batchsize * self.future_frames,6)).view(batchsize,self.future_frames,3,3)
        abs_hand_rot = matrix_to_rotation_6d(delta_hand_rot @ rotation_6d_to_matrix(current_hand_qpose[:,3:9]).unsqueeze(1)) # [batchsize,future_frames,6]
        abs_hand_trans = delta_hand_full_qpose[...,:3] + current_hand_qpose[:,:3].unsqueeze(1)# add the wind dim [batchsize,future_frames,3]
        abs_hand_joint_angle = delta_hand_full_qpose[...,9:] + current_hand_qpose[:,9:].unsqueeze(1)# add the wind dim [batchsize,future_frames,21]
        abs_hand_full_qpose = torch.cat([abs_hand_trans,abs_hand_rot,abs_hand_joint_angle],dim = -1)


        model_predict_res = {
            "abs_hand_full_qpose":abs_hand_full_qpose,# []

            "batch_current_target_qpose": batch_current_target_qpose,
            "batch_final_qpose_surface_points":batch_final_qpose_surface_points
        }

        return model_predict_res

    def forward(self, data):
        '''
        param:
            wind_qpose: [batchsize,wind_size,qpose_dim]
                but only need 22 dim, the wrist dim is not use
            wind_obj_pcd shape [batch_size,wind_size,pcd_num,3]

            data["other_item"]
        '''
        
        model_predict_res = self.sample(data)

        losses = self.cal_loss(model_predict_res,data)
        return losses



    def cal_loss(self,model_res,gt_data):
        '''
        these are 10 frames
        future_ten_frame_delta_hand_pcd: hand point change
        delta_rh2last: hand current to goal offset change
        '''
        
        batch_size = gt_data["wind_obj_pcd"].shape[0]
        future_slice = slice(self.past_frames,self.window_size)
        
        gt_future_hand_qpose = gt_data["wind_hand_qpose"][:,future_slice]
        
        losses = {}

        if self.use_qpose_hand_pcd_loss:
            model_predict_future_hand_qpose_surface_point = self.hand_model.get_surface_points(model_res["abs_hand_full_qpose"].view(batch_size * self.future_frames ,self.qpose_dim)).view(batch_size,self.future_frames,-1,3)
            gt_hand_future_qpose_surface_point = self.hand_model.get_surface_points(gt_future_hand_qpose.reshape(batch_size * self.future_frames,self.qpose_dim)).view(batch_size,self.future_frames,-1,3)
            losses["qpose_hand_pcd"] = self.LossL2(model_predict_future_hand_qpose_surface_point,gt_hand_future_qpose_surface_point) * 55 

        if self.use_qpose_loss:
            losses["qpose"] = self.LossL1(model_res["abs_hand_full_qpose"],gt_future_hand_qpose) *130
        

        loss = 0
        for value in losses.values():
            loss += value
        losses["loss"] = loss
        pprint(losses)

        return losses



