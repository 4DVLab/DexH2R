# import debugpy
# debugpy.listen(("localhost", 15000))
# debugpy.wait_for_client()  # 等待调试器连接


import os
import sys
sys.path.append(os.getcwd())
import torch
import numpy as np
from omegaconf import DictConfig
import random
from pytorch3d.transforms import rotation_6d_to_matrix
import hydra
from models.base import create_model
from tqdm import tqdm
from datasets.misc import collate_fn_general
from models.base import create_model
from datasets.base import create_dataset
from utils.e3m5_hand_model import get_e3m5_handmodel
from os.path import join as pjoin
import trimesh as tm
import colorama
from chamfer_distance import ChamferDistance as chamfer_dist
from pytorch3d.transforms import rotation_6d_to_matrix,matrix_to_euler_angles
from models.model.motion_net import find_nearest_final_qpose_using_fourty_trans_top_rotation

from pytorch3d.transforms import matrix_to_quaternion,quaternion_to_matrix,matrix_to_rotation_6d
from pytransform3d.rotations import quaternion_slerp
import colorama
import time





def load_ckpt(model: torch.nn.Module, ckpt_dir: str, pretrain_model_index: int): #-> (int, int):
    """ Load model and corresponding data

    Args:
        model: model to load the state dict
        ckpt_dir: directory where checkpoints are saved
        save_model_separately: flag indicating if checkpoints are saved separately

    Returns:
        epoch: last epoch
        step: last step
    """
    print(colorama.Fore.RED +  "warning!!!! the model is using the pretrain weight")
    ckpt_path = os.path.join(ckpt_dir, f"model_{pretrain_model_index}.pth")

    print("-" * 100)
    print("load model from ckpt")
    print("-"*100)
    print(ckpt_path)
    checkpoint = torch.load(ckpt_path)
    # 创建新的state字典，移除"module."前缀
    new_state_dict = {}
    for k, v in checkpoint["model"].items():
        name = k.replace("module.", "") # 移除"module."前缀
        new_state_dict[name] = v


    model.load_state_dict(new_state_dict)
    
    epoch = checkpoint['epoch']
    step = checkpoint['step']
    # return epoch, step

def load_item(data_path):
    data = np.load(data_path,allow_pickle=True)[()]
    data = torch.from_numpy(data).to("cuda").to(torch.float)[...,:3]# only need the xyz
    return data




def move_dict_data_to_device(data,device):
    for key in data:
        if torch.is_tensor(data[key]):
            data[key] = data[key].to(device)
    return data

def add_repeat_tail_data(seq_data,padding_num = 100):
    tail_seq_pcd_data = seq_data["seq_pcd"][-1].unsqueeze(0).expand(padding_num, -1,-1)
    seq_data["seq_pcd"] = torch.cat([seq_data["seq_pcd"],tail_seq_pcd_data],dim = 0)

    tail_seq_final_qpose_data = seq_data["seq_final_qpose_pool"][-1].unsqueeze(0).expand(padding_num, -1,-1)
    seq_data["seq_final_qpose_pool"] = torch.cat([seq_data["seq_final_qpose_pool"],tail_seq_final_qpose_data],dim = 0)

    tail_seq_mesh_model_surface_pcd_data = seq_data["obj_meshodel_surface_pcd"][-1].unsqueeze(0).expand(padding_num, -1,-1)
    seq_data["obj_meshodel_surface_pcd"] = torch.cat([seq_data["obj_meshodel_surface_pcd"],tail_seq_mesh_model_surface_pcd_data],dim = 0)

    return seq_data

def batch_hand_qpose_from_6d_rot_to_euler(batch_qpose):
    '''
    batch_qpose:[batch,qpose]
    '''
    euler_angle = matrix_to_euler_angles(rotation_6d_to_matrix(batch_qpose[:,3:9]))
    euler_angle_batch_qpose = torch.cat([batch_qpose[:,:3], euler_angle, batch_qpose[:,9:]],dim = -1)
    return euler_angle_batch_qpose


def slerp_quat(v1_,v2_,inter_num):
    '''
    param:
        v1_rot: [4,] torch.tensor
        v2_rot: [4,] torch.tensor
        inter_num: int
    return 
        slerp: [inter_num + 2,4]
    '''
    device = v1_.device
    v1 = v1_.cpu().numpy()
    v2 = v2_.cpu().numpy()
    inter_num += 2
    slerp = np.zeros((inter_num, 4), dtype=np.float32)
    
    interpolation_weight =np.linspace(0, 1, inter_num)
    for i, weight in enumerate(interpolation_weight):
        slerp[i] = quaternion_slerp(v1, v2, weight,shortest_path = True)
    return torch.from_numpy(slerp).to(device=device, dtype=v1_.dtype)




def interpolate_motion(v1, v2, _num_steps=20):
    '''
    param:
        v1,v2 need to
        because interpolation must be applied on quat,so the axis angle rotation will be changed to quat
    return:
        [_num_steps,qpos_dim]
    '''
    if v1.dim() == 1:   
        v1 = v1.unsqueeze(0)
    if v2.dim() == 1:
        v2 = v2.unsqueeze(0)

    rotation_part = slice(3,9)
    translation_part = slice(3)
    joint_angle_part = slice(9,None)

    interpolate_motion = torch.zeros(_num_steps + 2,v1.shape[-1]).to(v1.device)
    v1_quat = matrix_to_quaternion(rotation_6d_to_matrix(v1[...,rotation_part])).squeeze(0)
    v2_quat = matrix_to_quaternion(rotation_6d_to_matrix(v2[...,rotation_part])).squeeze(0)
    interpolate_motion[...,3:9] =  matrix_to_rotation_6d(quaternion_to_matrix(slerp_quat(v1_quat,v2_quat,_num_steps)))
    weights = torch.linspace(0, 1, _num_steps + 2, dtype=torch.float).view(-1, 1).to("cuda" if torch.cuda.is_available() else "cpu")
    interpolate_motion[...,translation_part] = (1 - weights) * v1[...,translation_part] + weights * v2[...,translation_part]    
    interpolate_motion[...,joint_angle_part] = (1 - weights) * v1[...,joint_angle_part] + weights * v2[...,joint_angle_part]    

    return interpolate_motion





def read_path_file(file_path):
    path_list = []
    
    # 使用 with 语句打开文件，自动处理文件关闭
    with open(file_path, 'r') as f:
        # 逐行读取文件内容
        for line in f:
            # 去除每行末尾的换行符并添加到列表中
            path_list.append(line.strip())
    
    return path_list




@hydra.main(version_base=None, config_path="./../configs", config_name="motion")
def main(cfg: DictConfig) -> None:

    # itp_mode = "10cm"
    itp_mode = f"{cfg.task.itp_mode}cm"


    #load model 
    device = f'cuda'

    model = create_model(cfg, device=device)
    model.to(device=device)
    assert cfg.use_pretrain == True, "not using the ckpt"
    load_ckpt(model, cfg.pretrain_model_dir_path,cfg.pretrain_model_index)
    model.eval()
    # load model end

    hand_model = get_e3m5_handmodel("cuda")
    chd = chamfer_dist()
    past_frames = cfg.task.dataset.past_frames
    use_predict_num = cfg.model.use_predict_num

    test_dataset = create_dataset(cfg.task.dataset, cfg.task.eval_task)

    seq_data_len = test_dataset.get_seq_len()
    # seq_data_len = 50
    
    result = []


    with torch.no_grad():  
        for seq_idx in tqdm(torch.arange(seq_data_len),desc = "seq_data"):

            seq_data = test_dataset.get_full_seq_data(seq_idx)

            gtdata_dir_path = seq_data["gtdata_dir_path"]

            print("original seq_pcd seq_len",seq_data["seq_pcd"].shape[0])
            seq_data = move_dict_data_to_device(seq_data,device)

            # one seq init
            seq_len = seq_data["seq_final_qpose_pool"].shape[0]

            curent_data_index = past_frames - 2

            model_predict_traj_list = [seq_data["seq_qpose"][idx] for idx in torch.arange(past_frames)]
            infer_frame_flag_list = []
            infer_method_flag_list = []
            model_select_final_pose = []
            every_frame_final_pose = []
            success_flag = False

            skip_frame = 0
            interpolation_threshold = 0

            while curent_data_index < seq_len -1 :
                # print("now index ",curent_data_index)
                curent_data_index += 1


                history_slice = slice(curent_data_index - past_frames + 1 ,curent_data_index + 1)# Because it's closed on the left and open on the right
                # current_hand_qpose = seq_data["seq_qpose"][curent_data_index]
                current_hand_qpose = model_predict_traj_list[-1]
                current_hand_surface_pcd = hand_model.get_surface_points(current_hand_qpose.unsqueeze(0))# [1,pcd_num,3]
                current_obj_obspcd = seq_data["seq_pcd"][curent_data_index]
                current_obj_mesh_pcd = seq_data["obj_meshodel_surface_pcd"][curent_data_index]
                last_frame_obj_mesh_pcd = seq_data["obj_meshodel_surface_pcd"][curent_data_index - 1]

                current_final_qpose_pool = seq_data["seq_final_qpose_pool"][curent_data_index]
                history_final_qpose_pool = seq_data["seq_final_qpose_pool"][history_slice]

                current_target_qpose,current_target_qpose_rot_diff,current_target_qpose_trans_diff = find_nearest_final_qpose_using_fourty_trans_top_rotation(current_hand_qpose.unsqueeze(0),current_final_qpose_pool.unsqueeze(0))
                current_target_qpose_rot_diff = current_target_qpose_rot_diff.squeeze(0)
                current_target_qpose_trans_diff = current_target_qpose_trans_diff.squeeze(0)
                current_target_qpose = current_target_qpose.squeeze(0)

                if len(model_select_final_pose) == 0:
                    model_select_final_pose.extend([current_target_qpose for _ in np.arange(5)])
                    infer_frame_flag_list.extend([True] * 5)
                    every_frame_final_pose.extend([current_target_qpose for _ in np.arange(5)])
                else:
                    every_frame_final_pose.append(current_target_qpose)

                if skip_frame != 0:
                    skip_frame -= 1
                    continue



                if len(model_predict_traj_list) > 50:

                    obj_pcd_distance_diff, obj2hand_dist, idx1, idx2 = chd(last_frame_obj_mesh_pcd.unsqueeze(0),current_obj_mesh_pcd.unsqueeze(0))# [1,pcd_num]
                    obj_pcd_distance_diff = obj_pcd_distance_diff.sum()
                    global_pose_changes = current_target_qpose_trans_diff * 100 # + #current_target_qpose_rot_diff  
                    if itp_mode == "5cm":
                        if (global_pose_changes < 5 and obj_pcd_distance_diff != 0) or interpolation_threshold:
                            interpolation_threshold = global_pose_changes / 2
                    else:
                        itp_mode == "10cm"
                        if global_pose_changes < 10 or interpolation_threshold:
                                interpolation_threshold = global_pose_changes / 2

                if interpolation_threshold:#if the distance between the hand and the obj is lower than 10cm, then interpolation
                    # when model predict length is more than 45 and distan ....
                    infer_method_flag_list.append(False)
                    infer_frame_flag_list.append(False)
                    model_select_final_pose.append(current_target_qpose)
                    print(colorama.Fore.RED + f"interpolate frame: {len(model_predict_traj_list)}")
                    interpolation_num = torch.floor(interpolation_threshold).int().item()

                    inter_motion = interpolate_motion(current_hand_qpose,current_target_qpose,interpolation_num)# interpolation_num + 2

                    if interpolation_num > 1:
                        model_predict_traj_list.append(inter_motion[1]) # the qose add need to be in [qpose_dim] shape 
                    else:
                        model_predict_traj_list.append(current_target_qpose)
                        success_flag = True # if the hand can touch the final pose, so the motion is successful
                        break
                else:
                    print(f"model predict in seq_idx {seq_idx}, from {len(model_predict_traj_list)} to {len(model_predict_traj_list) + use_predict_num}" )
                    skip_frame = use_predict_num - 1# because curent_data_index will add use_predict_num frames 
                    history_hand_qpose = torch.stack(model_predict_traj_list[- past_frames:])
                    history_obj_obspcd = seq_data["seq_pcd"][history_slice]
                    model_input_data = {
                        "wind_obj_pcd":history_obj_obspcd,
                        "wind_hand_qpose":history_hand_qpose,
                        "final_qpose_pool":history_final_qpose_pool
                    }
                    for key in model_input_data:
                        model_input_data[key] = model_input_data[key].unsqueeze(0) # add batch size

                    # extract the batch 0 data,because this method only need batchsize == 1
                    start_time = time.time()
                    model_predict = model.sample(model_input_data)
                    end_time = time.time()
                    print(f"time cost: {end_time - start_time} seconds")

                    model_predict_traj = model_predict["abs_hand_full_qpose"][0][:use_predict_num] # ？？？？？？？这个 seq_len 好像是错了
                    motion_adding_list = [model_predict_traj[idx] for idx in torch.arange(use_predict_num)]
                    model_predict_traj_list.extend(motion_adding_list)

                    infer_frame_flag_list.extend(([True] * use_predict_num))
                    infer_method_flag_list.append(True)

                    model_select_final_pose.extend([current_target_qpose for _ in np.arange(use_predict_num)])

            
            model_predict_traj_list = [item.to("cpu") for item in model_predict_traj_list]
            model_predict_traj_tensor = torch.stack(model_predict_traj_list)[:seq_len]
            infer_frame_flag_list = infer_frame_flag_list[:seq_len]
            result.append({
                "model_predict_traj_tensor":model_predict_traj_tensor,
                "infer_frame_flag_list":infer_frame_flag_list,
                "infer_method_flag_list":infer_method_flag_list,
                "model_select_final_pose":torch.stack(model_select_final_pose)[:seq_len],
                "success_flag":success_flag,
                "gtdata_dir_path":gtdata_dir_path,
                "every_frame_final_pose":every_frame_final_pose
            })

    data_save_dir_path = f"{cfg.viz_motion_output_dir}/{cfg.task.eval_task}"
    os.makedirs(data_save_dir_path,exist_ok=True)
    torch.save(result,pjoin(data_save_dir_path,f"vis_data.pth"))


if __name__ == '__main__':
    ## set random seed
    colorama.init(autoreset=True) 
    seed = 0
    torch.backends.cudnn.benchmark = False     
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    main()



# 