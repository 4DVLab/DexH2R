from typing import Dict
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.dataset.base_dataset import BaseDataset

class BaseRunner:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def run(self, policy: BasePolicy) -> Dict:
        raise NotImplementedError()

    def run_ours(self, policy: BasePolicy, test_dataloader) -> Dict:
        pass

    def visual(self, policy: BasePolicy, val_dataloader) -> Dict:
        # 这里可以使用 dataset 进行可视化
        pass