import wandb
import numpy as np
import torch
import collections
import tqdm
from diffusion_policy_3d.env import MetaWorldEnv
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper
import os
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util
from termcolor import cprint
from typing import Dict
import time
import math
from diffusion_policy_3d.env_runner.e3m5_hand_model import hand_loss, get_e3m5_handmodel,Evaluator , dis_loss, pen_loss
from pytorch3d.loss import chamfer_distance
import trimesh
class FastGraspEnvRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=1000,
                 n_obs_steps=8,
                 n_action_steps=8,
                 horizon = 16,
                 fps=10,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 n_envs=None,
                 task_name=None,
                 n_train=None,
                 n_test=None,
                 device="cuda:0",
                 use_point_crop=True,
                 num_points=512,
                 velocity_as_obs=False,
                 finalgrasp_as_obs=False,
                 ):
        super().__init__(output_dir)
        self.task_name = task_name
        self.eval_episodes = eval_episodes
        self.tqdm_interval_sec = tqdm_interval_sec
        if n_obs_steps + n_action_steps <= horizon:
            self.n_action_steps = n_action_steps
        else:
            self.n_action_steps = horizon - n_obs_steps
        self.n_obs_steps = n_obs_steps
        self.velocity_as_obs = velocity_as_obs
        self.finalgrasp_as_obs = finalgrasp_as_obs
        cprint(f"eval:finalgrasp_as_obs: {self.finalgrasp_as_obs}", 'magenta')
        cprint(f"eval:velocity_as_obs: {self.velocity_as_obs}", 'magenta')
        cprint(f"eval:n_obs_steps: {self.n_obs_steps}", 'magenta')
        cprint(f"eval:n_action_steps: {self.n_action_steps}", 'magenta')
        cprint(f"eval:horizon: {horizon}", 'magenta')
    def run(self, policy: BasePolicy) -> Dict:
        return dict()
    
    def log_and_store_metrics(self, metrics, all_metrics_lists,gt_total_pener_frame,gt_max_pener_depth,Is_interpolate):
        """
        Log metrics to WandB and store them into corresponding lists.
        
        Args:
            metrics (dict): 当前轨迹的指标字典。
            all_metrics_lists (dict): 保存所有轨迹指标的字典，键为 WandB 日志中的 key,值为列表。
        """
        metrics["total_pener_frame"] = max(0, metrics["total_pener_frame"]-gt_total_pener_frame)
        metrics["max_pener_depth"] = max(0, metrics["max_pener_depth"]-gt_max_pener_depth*(metrics["total_infer_frame"]-metrics["itp_infer_frame"]))
        metrics["dp_safety_rate"] = 0 if metrics["total_pener_frame"] > 0 else 1
        # metrics["dp_success_rate"] = 1 if Is_traj_end else 0
        metrics["dp_success_rate"] = 1 if Is_interpolate else 0
        # WandB log
        wandb_metrics = {
            "total_infer_frame": metrics["total_infer_frame"],
            "inference_once_time": metrics["total_infer_time"] / metrics["total_infer_frame"],
            "inference_total_time": metrics["total_infer_time"],
            "traj_accum_length": metrics["traj_accum_length"],
            "max_pener_depth": metrics["max_pener_depth"] / metrics["total_infer_frame"],
            "total_pener_frame": metrics["total_pener_frame"],
            "dp_safety_rate": metrics["dp_safety_rate"],
            "dp_success_rate": metrics["dp_success_rate"],
            "itp_infer_rate": metrics["itp_infer_frame"] / metrics["total_infer_frame"] 
        }
        wandb.log(wandb_metrics)

        # Store metrics into lists
        for key, value in wandb_metrics.items():
            if key in all_metrics_lists:
                all_metrics_lists[key].append(value)



    def run_ours(self, policy: BasePolicy, test_dataloader) -> Dict:
        device = policy.device       
        # begin evaluating
        with torch.no_grad():
            with tqdm.tqdm(test_dataloader, desc=f"Validation", leave=False, mininterval=1) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    processed_data= evaluator.preprocess_batch(batch, device, self.velocity_as_obs, self.finalgrasp_as_obs)
                    traj_index = processed_data["traj_index"]
                    agent_pos = processed_data["agent_pos"]
                    gt_actions = processed_data["gt_actions"]
                    objpcd_intact = processed_data["objpcd_intact"]
                    objpcd_normal_intact = processed_data["objpcd_normal_intact"]
                    velocity = processed_data["velocity"]
                    final_grasp = processed_data["final_grasp"]
                    final_grasp_group = processed_data["final_grasp_group"]
                    obs_objpcd = processed_data["obs_objpcd"]
                    ## validation里第一个traj的obs初始化：
                    if metrics["total_infer_frame"] == 0:
                        agent_pos_list = [agent_pos[i,:].unsqueeze(0) for i in range(self.n_obs_steps)] # element shape: [1, 28]  len = 8
                        final_grasp_list = [final_grasp[i,:].unsqueeze(0) for i in range(self.n_obs_steps)] if self.finalgrasp_as_obs else None
                        velocity_list = [velocity[i,:].unsqueeze(0) for i in range(self.n_obs_steps)] if self.velocity_as_obs else None
                        last_traj_index = traj_index  # initialize last_traj_idx
                        last_agent_trans = agent_pos_list[self.n_obs_steps-1][0,:3]
                    # 说明属于这个traj的batch已经结束了，可以开始统计metric了
                    if traj_index != last_traj_index:
                        self.log_and_store_metrics(metrics, all_metrics_lists, gt_total_pener_frame, gt_max_pener_depth,Is_interpolate)
                        if Is_visualize and Is_interpolate:
                            len_dp_data = len(visualizations["pred_action"])
                            len_itp_data = len(visualizations["itp_action"])
                            for i in range(len_dp_data):
                                pred_hand_mesh = visualizations["pred_action"][i]
                                gt_hand_mesh = visualizations["gt_action"][i]
                                final_grasp_mesh = visualizations["final_grasp"][i]
                                object_mesh = visualizations["object"][i]
                                pred_hand_mesh.export(f'{sub_folder_paths["pred_action"]}/pred_hand_{i}.ply')
                                gt_hand_mesh.export(f'{sub_folder_paths["gt_action"]}/gt_hand_{i}.ply')
                                final_grasp_mesh.export(f'{sub_folder_paths["final_grasp"]}/final_grasp_{i}.ply')
                                object_mesh.export(f'{sub_folder_paths["object"]}/object_{i}.ply')
                            for i in range(len_itp_data):
                                pred_hand_mesh = visualizations["itp_action"][i]
                                gt_hand_mesh = visualizations["itp_gt_action"][i]
                                final_grasp_mesh = visualizations["itp_final_grasp"][i]
                                object_mesh = visualizations["itp_object"][i]
                                pred_hand_mesh.export(f'{sub_folder_paths["itp_action"]}/itp_hand_{i}.ply')
                                gt_hand_mesh.export(f'{sub_folder_paths["itp_gt_action"]}/itp_gt_hand_{i}.ply')
                                final_grasp_mesh.export(f'{sub_folder_paths["itp_final_grasp"]}/itp_final_grasp_{i}.ply')
                                object_mesh.export(f'{sub_folder_paths["itp_object"]}/itp_object_{i}.ply')
                        traj_num += 1
                        traj_folder = f"{eval_folder}/traj_{traj_num}"
                        sub_folder_paths = evaluator.create_folders(traj_folder, sub_folder_names)
                        Is_traj_end = False
                        Is_interpolate = False
                        Interpolate_begin = False
                        agent_pos_list = [agent_pos[i,:].unsqueeze(0) for i in range(self.n_obs_steps)]
                        velocity_list = [velocity[i,:].unsqueeze(0) for i in range(self.n_obs_steps)] if self.velocity_as_obs else None
                        final_grasp_list = [final_grasp[i,:].unsqueeze(0) for i in range(self.n_obs_steps)] if self.finalgrasp_as_obs else None
                        last_traj_index = traj_index
                        last_agent_trans = agent_pos_list[self.n_obs_steps-1][0,:3]
                        metrics = evaluator._initialize_metrics()
                        visualizations = {
                            "pred_action": [],
                            "gt_action": [],
                            "final_grasp": [],
                            "object": [],
                            "itp_action": [],
                            "itp_gt_action": [],
                            "itp_final_grasp": [],
                            "itp_object": []
                        }
                    # 说明还有batch包含这个traj的信息
                    elif traj_index == last_traj_index:
                        # 说明这个traj已经提前结束了，手提前抓到了物体 -> 好； 那么跳过后面的batch, 直到traj_index != last_traj_index
                        if Is_traj_end:
                            last_traj_index = traj_index
                            continue
                        # 说明这个traj还没有结束，需要继续infer; 而infer的方式取决于Is_interpolate这个变量！
                        else:
                            if Interpolate_begin:
                                # 只需要考虑每个batch的第一个pos, 因为每个batch只往后推动一个obs
                                curr_obs_index = self.n_obs_steps - 1
                                curr_hand_pose = evaluator.insert_zeros(inter_begin_pose) # [1, 28]->[1, 30] 为上一次dp infer的最后一个hand, 也就是新更新的agent_pos_list里的第n_obs_steps个
                                curr_objpcd_intact = objpcd_intact[curr_obs_index] # [4096, 3]
                                curr_objpcd_normal_intact = objpcd_normal_intact[curr_obs_index] # [4096, 3]
                                curr_gt_hand_pose = evaluator.insert_zeros(gt_actions[curr_obs_index].unsqueeze(0)) # [28] -> [1, 28] -> [1, 30]
                                curr_final_grasp_group = evaluator.insert_zeros(final_grasp_group[curr_obs_index]) # [500, 28] = [final_grasp_num, 28] -> [500, 30]
                                # TODO: 根据curr_hand_pose和curr_final_grasp_group，选取一个final_grasp, 并返回
                                curr_final_grasp, _ = evaluator.find_nearest_final_qpose(curr_hand_pose, curr_final_grasp_group.unsqueeze(0)) # [1, 30]
                                # TODO: 先计算curr_hand_pos和curr_final_grasp之间的距离（用compute_pose_distance）,然后根据这个下取整得到num_interpolate_steps
                                itp_num_steps = math.floor(evaluator.compute_pose_distance(curr_hand_pose.squeeze(0), curr_final_grasp.squeeze(0)) / 2)
                                # print("itp_num_steps = ", itp_num_steps)
                                # TODO: 如果itp_num_steps == 0, 那么is_traj_end = True
                                if itp_num_steps == 0:
                                    Is_traj_end = True
                                # TODO: 根据curr_hand_pose和final_grasp进行inteprolate, 返回第一个intepolation
                                interpolate_hand = evaluator.interpolate_motion(curr_hand_pose, curr_final_grasp, itp_num_steps)[1].unsqueeze(0) # [1, 30] 因为index=0是起点的hand pose
                                # TODO: 将这个第一个interpolation hand mesh 存到itp_action list里
                                interpolate_hand_mesh = self.hand_model.get_meshes_from_q(interpolate_hand, i=0)
                                visualizations["itp_action"].append(interpolate_hand_mesh)
                                # TODO: 将curr_gt_hand_pose mesh存进itp_gt_hand list
                                curr_gt_hand_pose_mesh = self.hand_model.get_meshes_from_q(curr_gt_hand_pose, i=0)
                                visualizations["itp_gt_action"].append(curr_gt_hand_pose_mesh)
                                # TODO: 将curr_final_grasp mesh存进itp_final_grasp list
                                curr_final_grasp_mesh = self.hand_model.get_meshes_from_q(curr_final_grasp, i=0)
                                visualizations["itp_final_grasp"].append(curr_final_grasp_mesh)
                                # TODO: 将objpcd_intact mesh存进itp_object list
                                curr_objpcd_intact_points = curr_objpcd_intact.cpu().numpy()
                                curr_objpcd_intact_mesh = trimesh.points.PointCloud(curr_objpcd_intact_points)
                                visualizations["itp_object"].append(curr_objpcd_intact_mesh)
                                # TODO: 计算interpolation hand 以及 final_grasp_hand的translation之间的distance, 更新metric中的traj_accum_length
                                metrics["traj_accum_length"] += torch.sqrt(torch.sum((curr_hand_pose[0][:3] - interpolate_hand[0][:3])**2))
                                # TODO: 更新metrics中的itp_infer_frame
                                metrics["itp_infer_frame"] += 1
                                # TODO: 更新下一次interpolate的起始位置
                                inter_begin_pose = evaluator.remove_zeros(interpolate_hand)
                            if metrics["total_infer_frame"] % self.n_action_steps == 0 and Interpolate_begin == False:
                                if Is_interpolate:
                                    # import pdb; pdb.set_trace()
                                    # 只需要考虑每个batch的第一个pos, 因为每个batch只往后推动一个obs
                                    curr_obs_index = self.n_obs_steps - 1
                                    curr_hand_pose = evaluator.insert_zeros(inter_begin_pose) # [1, 28]->[1, 30] 为上一次dp infer的最后一个hand, 也就是新更新的agent_pos_list里的第n_obs_steps个
                                    curr_objpcd_intact = objpcd_intact[curr_obs_index] # [4096, 3]
                                    curr_objpcd_normal_intact = objpcd_normal_intact[curr_obs_index] # [4096, 3]
                                    curr_gt_hand_pose = evaluator.insert_zeros(gt_actions[curr_obs_index].unsqueeze(0)) # [28] -> [1, 28] -> [1, 30]
                                    curr_final_grasp_group = evaluator.insert_zeros(final_grasp_group[curr_obs_index]) # [16, 500, 28] -> [500, 28] = [final_grasp_num, 28] -> [500, 30]
                                    # TODO: 根据curr_hand_pose和curr_final_grasp_group，选取一个final_grasp, 并返回
                                    curr_final_grasp, _ = evaluator.find_nearest_final_qpose(curr_hand_pose, curr_final_grasp_group.unsqueeze(0)) # [1, 30]
                                    # TODO: 先计算curr_hand_pos和curr_final_grasp之间的距离（用compute_pose_distance）,然后根据这个下取整得到num_interpolate_steps
                                    itp_num_steps = math.floor(evaluator.compute_pose_distance(curr_hand_pose.squeeze(0), curr_final_grasp.squeeze(0)) / 2)
                                    # print("itp_num_steps = ", itp_num_steps)
                                    if itp_num_steps == 0:
                                        Is_traj_end = True
                                    # TODO: 根据curr_hand_pose和final_grasp进行inteprolate, 返回第一个intepolation
                                    interpolate_hand = evaluator.interpolate_motion(curr_hand_pose, curr_final_grasp, itp_num_steps)[1].unsqueeze(0) # [1, 30] 因为index=0是起点的hand pose
                                    # TODO: 将这个第一个interpolation hand mesh 存到itp_action list里
                                    interpolate_hand_mesh = self.hand_model.get_meshes_from_q(interpolate_hand, i=0)
                                    visualizations["itp_action"].append(interpolate_hand_mesh)
                                    # TODO: 将curr_gt_hand_pose mesh存进itp_gt_hand list
                                    curr_gt_hand_pose_mesh = self.hand_model.get_meshes_from_q(curr_gt_hand_pose, i=0)
                                    visualizations["itp_gt_action"].append(curr_gt_hand_pose_mesh)
                                    # TODO: 将curr_final_grasp mesh存进itp_final_grasp list
                                    curr_final_grasp_mesh = self.hand_model.get_meshes_from_q(curr_final_grasp, i=0)
                                    visualizations["itp_final_grasp"].append(curr_final_grasp_mesh)
                                    # TODO: 将objpcd_intact mesh存进itp_object list
                                    curr_objpcd_intact_points = curr_objpcd_intact.cpu().numpy()
                                    curr_objpcd_intact_mesh = trimesh.points.PointCloud(curr_objpcd_intact_points)
                                    visualizations["itp_object"].append(curr_objpcd_intact_mesh)
                                    # TODO: 计算interpolation hand 以及 objpcd之间的max_pener_depth, total_pener_frame更新metric
                                    #import pdb; pdb.set_trace()
                                    # itp_hand_pcd = self.hand_model.get_surface_points(q=interpolate_hand).to(dtype=torch.float32)
                                    # curr_obj_pcd_nor = torch.cat((curr_objpcd_intact.unsqueeze(0), curr_objpcd_normal_intact.unsqueeze(0)), dim=-1).to(dtype=torch.float32)
                                    # pen_loss_value = pen_loss(curr_obj_pcd_nor, itp_hand_pcd)
                                    # metrics["max_pener_depth"] += pen_loss_value
                                    # metrics["total_pener_frame"] += (1 if pen_loss_value > 0 else 0)
                                    # TODO: 计算interpolation hand 以及 final_grasp_hand的translation之间的distance, 更新metric中的traj_accum_length
                                    metrics["traj_accum_length"] += torch.sqrt(torch.sum((curr_hand_pose[0][:3] - interpolate_hand[0][:3])**2))
                                    # TODO: 更新metrics中的itp_infer_frame
                                    metrics["itp_infer_frame"] += 1
                                    # TODO: 将interpolate_begin设置为true
                                    Interpolate_begin = True
                                    # TODO: 更新下一次interpolate的起始位置
                                    inter_begin_pose = evaluator.remove_zeros(interpolate_hand)
                                # 这个时候 dp infer
                                else:
                                    # TODO: 使用agent_pos_list, final_grasp_list以及其他原始的Obs更新obs
                                    new_obs_dict = evaluator.update_obs_dict(batch["obs"],self.n_obs_steps, agent_pos_list, velocity_list, final_grasp_list)
                                    # TODO: 将这个obs传入model进行infer,得到index=i的pred_hand, 因为["action"]已经size=n_action_steps了，并且已经是从horizon中的self.n_obs_steps开始取的了，所以这里index直接从0开始就可以
                                    # TODO: 计算model infer时间 更新metrics中的total_infer_time，once_infer_time
                                    start_time = time.time()
                                    pred_actions = policy.predict_action(new_obs_dict)["action"].squeeze(0) # [b, n_action_steps, 28] = [1, n_action_steps, 28] -> [n_action_steps, 28]
                                    end_time = time.time()
                                    metrics["total_infer_time"] += (end_time - start_time)
                                    # TODO: 找到每一个infer_hand对应的最近的final_grasp, final_grasp的index是从self.n_obs_steps开始
                                    obs_start_index = self.n_obs_steps
                                    pred_actions_30 = evaluator.insert_zeros(pred_actions) # [n_action_steps, 28] -> [n_action_steps, 30]
                                    chosen_final_grasp_30, _ = evaluator.find_nearest_final_qpose(pred_actions_30, evaluator.insert_zeros_group(final_grasp_group[obs_start_index:obs_start_index+self.n_action_steps])) # [n_action_steps, 30]
                                    # TODO: 每个pred_hand与objpcd算一个max_pener_depth以及total_pener_frame 更新metrics
                                    # TODO: metric: traj_accum_length
                                    for i in range(self.n_action_steps):
                                        curr_agent_trans = pred_actions[i][:3]
                                        metrics["traj_accum_length"] += torch.sqrt(torch.sum((curr_agent_trans - last_agent_trans)**2))
                                        last_agent_trans = curr_agent_trans
                                        pred_hand_pcd = self.hand_model.get_surface_points(q=pred_actions_30[i].unsqueeze(0)).to(dtype=torch.float32)
                                        curr_objpcd_normal_intact = objpcd_normal_intact[obs_start_index+i]
                                        curr_objpcd_intact = objpcd_intact[obs_start_index+i]
                                        curr_obj_pcd_nor = torch.cat((curr_objpcd_intact.unsqueeze(0), curr_objpcd_normal_intact.unsqueeze(0)), dim=-1).to(dtype=torch.float32)
                                        pen_loss_value = pen_loss(curr_obj_pcd_nor, pred_hand_pcd)
                                        metrics["max_pener_depth"] += pen_loss_value
                                        metrics["total_pener_frame"] += (1 if pen_loss_value > 0 else 0)
                                        # TODO: 将pred_hand, gt_hand, objpcd_intact和final_grasp_mesh存进对应的list里
                                        curr_gt_hand_pose = evaluator.insert_zeros(gt_actions[obs_start_index+i].unsqueeze(0))
                                        curr_final_grasp = chosen_final_grasp_30[i].unsqueeze(0)
                                        pred_hand_mesh = self.hand_model.get_meshes_from_q(q=pred_actions_30[i].unsqueeze(0), i=0)
                                        curr_gt_hand_pose_mesh = self.hand_model.get_meshes_from_q(q=curr_gt_hand_pose, i=0)
                                        curr_final_grasp_mesh = self.hand_model.get_meshes_from_q(q=curr_final_grasp, i=0)
                                        curr_objpcd_intact_points = curr_objpcd_intact.cpu().numpy()
                                        curr_objpcd_intact_mesh = trimesh.points.PointCloud(curr_objpcd_intact_points)
                                        visualizations["pred_action"].append(pred_hand_mesh)
                                        visualizations["gt_action"].append(curr_gt_hand_pose_mesh)
                                        visualizations["final_grasp"].append(curr_final_grasp_mesh)
                                        visualizations["object"].append(curr_objpcd_intact_mesh)
                                    # TODO: 更新metrics total_infer_frame += self.n_action_steps（不用！这个必须在外层做）
                                    # TODO: 根据pred_hand的最后一个即Index = self.n_obs_steps + self.n_action_steps-1 找到此时最近的final grasp
                                    # TODO: 计算pred_hand最后一个与final grasp之间的距离，如果 < 10cm 那么Is_interpolate = True
                                    last_pred_hand = pred_actions_30[-1]
                                    last_final_grasp = chosen_final_grasp_30[-1]
                                    distance = evaluator.compute_pose_distance(last_pred_hand, last_final_grasp)
                                    # print("distance = ", distance)
                                    #Is_interpolate = True if distance < 10 else False
                                    cd_loss_value, _ = chamfer_distance(torch.tensor(objpcd_intact[obs_start_index+self.n_action_steps-1], dtype=torch.float).unsqueeze(0), torch.tensor(objpcd_intact[obs_start_index+self.n_action_steps-2], dtype=torch.float).unsqueeze(0), point_reduction='mean', batch_reduction='mean')
                                    if distance < interpolation_threshold and cd_loss_value != 0 and metrics["total_infer_frame"] > total_infer_frame_threshold:
                                        Is_interpolate = True
                                        inter_begin_pose = pred_actions[-1].unsqueeze(0) # [1, 28]
                                        print("success")
                                    # TODO: 如果 > 10cm 那么接下来的Batch继续dp infer
                                    
                                    # TODO: 更新agent_pos_list和final_grasp_list作为下一次的infer的obs
                                    #import pdb; pdb.set_trace()
                                    if self.n_action_steps>= self.n_obs_steps:
                                        agent_pos_list = [pred_actions[j].unsqueeze(0) for j in range(self.n_action_steps-self.n_obs_steps, self.n_action_steps)]
                                        final_grasp_list = [evaluator.remove_zeros(chosen_final_grasp_30)[j].unsqueeze(0) for j in range(self.n_action_steps-self.n_obs_steps, self.n_action_steps)] if self.finalgrasp_as_obs else None  
                                        velocity_list = [pred_actions[j].unsqueeze(0)-pred_actions[j-1].unsqueeze(0) for j in range(self.n_action_steps-self.n_obs_steps, self.n_action_steps)] if self.velocity_as_obs else None
                                    else:
                                        # 当 n_action_steps < n_obs_steps 时，需要保留之前的部分观察
                                        keep_old_obs = self.n_obs_steps - self.n_action_steps
                                        # 保留旧的观察的后 keep_old_obs 个
                                        agent_pos_list = agent_pos_list[-keep_old_obs:] 
                                        # 添加新预测的所有动作
                                        agent_pos_list.extend([pred_actions[j].unsqueeze(0) for j in range(self.n_action_steps)])
                                        
                                        if self.finalgrasp_as_obs:
                                            final_grasp_list = final_grasp_list[-keep_old_obs:]
                                            final_grasp_list.extend([evaluator.remove_zeros(chosen_final_grasp_30)[j].unsqueeze(0) for j in range(self.n_action_steps)])
                                        
                                        if self.velocity_as_obs:
                                            velocity_list = velocity_list[-keep_old_obs:]
                                            velocity_list.extend([pred_actions[j].unsqueeze(0)-pred_actions[j-1].unsqueeze(0) for j in range(self.n_action_steps)])
                    metrics["total_infer_frame"] += 1
                    last_traj_index = traj_index
                    # print("total_infer_frame = ", metrics["total_infer_frame"])   
        log_data = {
            f"mean_{key}": np.mean(
                [v.cpu().numpy() if isinstance(v, torch.Tensor) else v for v in value]
            ) if len(value) > 0 else 0  # 如果列表为空，返回 0
            for key, value in all_metrics_lists.items()
        }
        wandb.log(log_data)
        return log_data
    
    def visual(self, policy: BasePolicy, visualdata) -> Dict:
        device = policy.device

        all_traj_rewards = []
        all_success_rates = []
        hand_model = get_e3m5_handmodel(self.n_action_steps-1,"cuda")
        # for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Metaworld {self.task_name} Pointcloud Env", leave=False, mininterval=self.tqdm_interval_sec):
        for batch_idx, batch in enumerate(visualdata):
            if batch_idx % self.n_obs_steps != 0:  # 处理当前 batch 后跳过两个
                continue
            policy.reset()
            obs_dict = batch['obs']
            done = False
            import pdb
            # 在这里打断点
            # pdb.set_trace()
            # print(batch['obs']['agent_pos'].shape)
            while not done:
                with torch.no_grad():
                    obs_dict_input = {}
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud']
                    # 判断 np_action_dict 是否存在
                    if 'np_action_dict' not in locals():
                        print(1)
                        obs_dict_input['agent_pos'] = obs_dict['agent_pos']
                    else:
                        obs_dict_input['agent_pos'] = torch.cat((torch.tensor(np_action_dict['action_pred'][:, 2:, :]), obs_dict['agent_pos'][:, 2:, :]), dim=1)
                    import time ; start_time = time.time()
                    action_dict = policy.predict_action(obs_dict_input)
                    import time ; end_time = time.time()
                    print(f"Time taken for predict_action: {end_time - start_time:.4f} seconds")
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())
             


                pred_q_tensor = torch.tensor(np_action_dict['action_pred'][:, self.n_obs_steps:, :]).to(device)###self.n_obs_steps 是 2
                zeros_actions = torch.zeros(pred_q_tensor.shape[0], pred_q_tensor.shape[1], 2).to(device)
                pred_q_tensor = torch.cat([pred_q_tensor[:,:,:6], zeros_actions, pred_q_tensor[:,:,6:]], dim=2)

                agent_pos_tensor=torch.cat([obs_dict['agent_pos'][:,:self.n_obs_steps,:6].to(device), zeros_actions, obs_dict['agent_pos'][:,:self.n_obs_steps,6:].to(device)], dim=2)
                
                # 使用 for 循环保存两个手的模型
                for step in range(self.n_obs_steps):
                    # 保存 agent_pos 对应的手模型
                    hand_mesh_agent = hand_model.get_meshes_from_q(q=agent_pos_tensor[0, :, :], i=step)
                    hand_mesh_agent.export(f"/inspurfs/group/mayuexin/zym/based_diffusion_policy/3D-Diffusion-Policy/visual_data/hand_agent_{batch_idx+step}.ply")  # 保存为 PLY 文件，文件名包含步骤索引

                    infer_hand_mesh_i = hand_model.get_meshes_from_q(q=pred_q_tensor[0, :, :], i=step)
                    infer_hand_mesh_i.export(f"/inspurfs/group/mayuexin/zym/based_diffusion_policy/3D-Diffusion-Policy/visual_data/pred_hand_{batch_idx+step}.ply")
                    # 获取当前点云数据
                    point_cloud = obs_dict['point_cloud'][0, step, 6:].detach().cpu().numpy()  # 获取第 step 个点云数据并转换为 NumPy 数组
                    # point_cloud 现在是形状为 (4090, 3)

                    # 创建 PLY 文件的头部
                    ply_header = f"""ply
                format ascii 1.0
                element vertex {point_cloud.shape[0]}
                property float x
                property float y
                property float z
                end_header
                """

                    # 将点云数据转换为字符串格式
                    point_cloud_data = "\n".join(" ".join(map(str, point)) for point in point_cloud)

                    # 保存为 PLY 文件
                    with open(f"/inspurfs/group/mayuexin/zym/based_diffusion_policy/3D-Diffusion-Policy/visual_data/point_cloud_{batch_idx + step}.ply", "w") as f:
                        f.write(ply_header)
                        f.write(point_cloud_data + "\n")  # 确保最后有一个换行符
                # print('action:',action)
                done = True
        log_data = dict()

        return log_data