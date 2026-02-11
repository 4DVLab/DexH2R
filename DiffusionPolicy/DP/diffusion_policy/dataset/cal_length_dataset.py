from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

class cal_length_dataset(BaseImageDataset):
    def __init__(self,
            zarr_path, 
            traj_num,
            train_mask_start,
            train_mask_end,
            val_mask_start,
            val_mask_end,
            horizon=16,
            pad_before=1,
            pad_after=7,
            seed=42,
            val_ratio=0.02,
            max_train_episodes=None
            ):
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['action'])
        self.traj_num = traj_num
        self.train_mask_start = train_mask_start
        self.train_mask_end = train_mask_end
        self.val_mask_start = val_mask_start
        self.val_mask_end = val_mask_end
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        val_mask = np.zeros(self.traj_num, dtype=bool)
        val_mask[self.val_mask_start:self.val_mask_end] = True

        train_mask = np.zeros(self.traj_num, dtype=bool)
        train_mask[self.train_mask_start:self.train_mask_end] = True
        
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon, # 我们这里的Horizon就设为1
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_mask = np.zeros(self.traj_num, dtype=bool)
        val_mask[self.val_mask_start:self.val_mask_end] = True
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=val_mask
            )
        val_set.train_mask = val_mask
        return val_set
    
    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer    

    def __len__(self) -> int:
        return len(self.sampler)
    
    def _sample_to_data(self, sample):
        data = {
            'action': sample['action'].astype(np.float32),
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
    
def test():
    import os
    zarr_path = os.path.expanduser('grasp_data.zarr')
    dataset = GraspImageDataset(zarr_path, horizon=16)  

        