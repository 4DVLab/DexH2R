from typing import Dict
import torch
import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
import numpy as np
import copy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from termcolor import cprint
class FastgraspDataset(BaseDataset):
    def __init__(self,
                 zarr_path,
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 seed=42,
                 val_ratio=0.0,
                 max_train_episodes=None,
                 max_val_episodes=None,
                 max_test_episodes=None,
                 task_name=None,
                 velocity_as_obs=False,
                 finalgrasp_as_obs=False,
                 traj_num = None,
                 train_mask_start = None,
                 train_mask_end = None,
                 val_mask_start = None,
                 val_mask_end = None,
                 test_mask_start = None,
                 test_mask_end = None,
                 **kwargs
                 ):
        super().__init__()
        self.velocity_as_obs = velocity_as_obs
        self.finalgrasp_as_obs = finalgrasp_as_obs
        cprint(f"DATASET finalgrasp_as_obs: {self.finalgrasp_as_obs}", 'magenta')
        cprint(f"DATASET velocity_as_obs: {self.velocity_as_obs}", 'magenta')
        self.traj_num = traj_num
        self.train_mask_start = train_mask_start
        self.train_mask_end = train_mask_end
        self.val_mask_start = val_mask_start
        self.val_mask_end = val_mask_end
        self.test_mask_start = test_mask_start
        self.test_mask_end = test_mask_end
        self.task_name = task_name
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['objpcd_intact','agent_pos','velocity','action','objpcd_normal_intact','final_grasp','obs_objpcd','traj_index','final_grasp_group'])
        # self.replay_buffer = ReplayBuffer.copy_from_path(
        #     zarr_path, keys=['objpcd_intact','agent_pos','action'])
        # import pdb; pdb.set_trace()
        train_mask = np.zeros(self.traj_num, dtype=bool)
        train_mask[self.train_mask_start:self.train_mask_end] = True
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)
        print("train_mask_length = ",train_mask.shape)
        print("Number of True values in train_mask = ", np.sum(train_mask))
        val_mask = np.zeros(self.traj_num, dtype=bool)
        val_mask[self.val_mask_start:self.val_mask_end] = True
        val_mask = downsample_mask(
            mask=val_mask, 
            max_n=max_val_episodes, 
            seed=seed)
        print("val_mask_length = ",val_mask.shape)
        print("Number of True values in val_mask = ", np.sum(val_mask))
        test_mask = np.zeros(self.traj_num, dtype=bool)
        test_mask[self.test_mask_start:self.test_mask_end] = True
        test_mask = downsample_mask(
            mask=test_mask, 
            max_n=max_test_episodes, 
            seed=seed)
        print("test_mask_length = ",test_mask.shape)
        print("Number of True values in test_mask = ", np.sum(test_mask))
        self.sampler = SequenceSampler(replay_buffer=self.replay_buffer,
                                        sequence_length=horizon,
                                        pad_before=pad_before,
                                        pad_after=pad_after,
                                        episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.val_mask = val_mask
        self.test_mask = test_mask

    def get_validation_dataset(self):
        val_set = copy.copy(self)

        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=self.val_mask
            )
        val_set.train_mask = self.val_mask
        return val_set
    
    def get_test_dataset(self):
        test_set = copy.copy(self)

        test_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=self.test_mask
            )
        test_set.train_mask = self.test_mask
        return test_set
    
    def get_normalizer(self, mode='limits', **kwargs):
        # 这里我们没有agent_pos,所以删除了data中的'agent_pos'
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['agent_pos'],
            # 'velocity': self.replay_buffer['velocity'],
            'point_cloud': self.replay_buffer['objpcd_intact'],
            # 'final_grasp': self.replay_buffer['final_grasp'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer
    
    def __len__(self) -> int:
        return len(self.sampler)
                
    def _sample_to_data(self, sample):

        agent_pos = sample['agent_pos'].astype(np.float32)
        velocity = sample['velocity'].astype(np.float32)
        obs_objpcd = sample['objpcd_intact'].astype(np.float32)
        final_grasp = sample['final_grasp'].astype(np.float32)
        final_grasp_group = sample['final_grasp_group'].astype(np.float32)
        objpcd_intact = sample['objpcd_intact'].astype(np.float32)
        objpcd_normal_intact = sample['objpcd_normal_intact'].astype(np.float32)
        traj_index = sample["traj_index"].astype(np.float32)
        data = {
            'obs': {
                'agent_pos': agent_pos,
                'point_cloud': objpcd_intact,
            },
            'action': sample['action'].astype(np.float32),
            'objpcd_intact': objpcd_intact,
            'objpcd_normal_intact': objpcd_normal_intact,
            'final_grasp_group': final_grasp_group,
            'final_grasp': final_grasp,
            'traj_index': traj_index,
            'obs_objpcd': obs_objpcd,
            'velocity': velocity
        }
        # if self.velocity_as_obs:
        #     data['obs']['velocity'] = velocity
        # else:
        #     data["velocity"] = velocity
        # if self.finalgrasp_as_obs:
        #     data['obs']['final_grasp'] = final_grasp
        # else:
        #     data['final_grasp'] = final_grasp
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

def main():
    dataset = FastgraspDataset(zarr_path='/inspurfs/group/mayuexin/wangyzh/DynamicGrasp/training_data/xiaochy_try/zarr_dataset/dp1_whole_final_grasp_diff.zarr', horizon=1, pad_before=0, pad_after=0, seed=42, val_ratio=0.0, max_train_episodes=None, task_name=None)
    print(len(dataset))
    print(dataset[15])

if __name__ == "__main__":
    main()
