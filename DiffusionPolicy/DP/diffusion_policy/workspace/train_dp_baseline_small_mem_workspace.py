if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
import gc
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.dataset.dp_baseline_dataset import dp_baseline_dataset
from diffusion_policy.dataset.eval_dp_dataset import eval_dp_dataset
from diffusion_policy.dataset.cal_length_dataset import cal_length_dataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.common.normalizer import LinearNormalizer

def delete_dataset(dataset):
    dataset.replay_buffer = None
    dataset.sampler = None
    del dataset
    gc.collect()

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionUnetHybridWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # configure model
        self.model: DiffusionUnetHybridImagePolicy = hydra.utils.instantiate(cfg.policy)
        self.ema_model: DiffusionUnetHybridImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)
        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())
        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        # resume training
        cfg.training.resume = False
        # if cfg.training.resume:
        if cfg.resume:
            lastest_ckpt_path = cfg.checkpoint_path
            print(f"Resuming from checkpoint {lastest_ckpt_path}")
            self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
    
        # Calculate len(train_dataloader) to compute training steps for lr_scheduler
        tmp_train_dataset = cal_length_dataset(cfg.task.train_dataset.zarr_path,
                                                  cfg.task.train_dataset.traj_num, cfg.task.train_dataset.train_mask_start, cfg.task.train_dataset.train_mask_end,
                                                  cfg.task.train_dataset.val_mask_start, cfg.task.train_dataset.val_mask_end,
                                                  pad_before=cfg.task.dataset.pad_before, pad_after=cfg.task.dataset.pad_after, horizon=cfg.task.dataset.horizon)
        tmp_train_dataloader = DataLoader(tmp_train_dataset, **cfg.dataloader)
        train_dataloader_length = len(tmp_train_dataloader)
        delete_dataset(tmp_train_dataset)
        
        # load the previously calculated normalizer into the memory
        normalizer = LinearNormalizer()
        normalizer.params_dict = torch.load(cfg.task.normalizer.path)
        

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)
        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                train_dataloader_length * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step-1
        )
        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)
        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )
        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )
        # device transfer
        device = torch.device(cfg.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        # save batch for sampling
        train_sampling_batch = None
        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        min_val_loss = float('inf')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                for trainset_num in range(1, cfg.training.subset_num+1):
                    prev_title = getattr(cfg.task, f"train_dataset_{trainset_num}")
                    train_dataset = dp_baseline_dataset(prev_title.zarr_path,
                                                prev_title.traj_num, prev_title.train_mask_start, prev_title.train_mask_end,
                                                prev_title.val_mask_start, prev_title.val_mask_end,
                                                pad_before=cfg.task.dataset.pad_before, pad_after=cfg.task.dataset.pad_after, horizon=cfg.task.dataset.horizon)
                    train_dataloader = DataLoader(train_dataset, **cfg.dataloader)
                    with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}; Train subset {trainset_num}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            # device transfer
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            if train_sampling_batch is None:
                                train_sampling_batch = batch
                            # compute loss
                            raw_loss = self.model.compute_loss(batch)
                            loss = raw_loss / cfg.training.gradient_accumulate_every
                            loss.backward()
                            # step optimizer
                            if self.global_step % cfg.training.gradient_accumulate_every == 0:
                                self.optimizer.step()
                                self.optimizer.zero_grad()
                                lr_scheduler.step()
                            # update ema
                            if cfg.training.use_ema:
                                ema.step(self.model)
                            # logging
                            raw_loss_cpu = raw_loss.item()
                            tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                            train_losses.append(raw_loss_cpu)
                            wandb.log({"train_loss": raw_loss_cpu})
                            step_log = {
                                'train_loss': raw_loss_cpu,
                                'global_step': self.global_step,
                                'epoch': self.epoch,
                                'lr': lr_scheduler.get_last_lr()[0]
                            }
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1
                        delete_dataset(train_dataset)

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss
                wandb.log({"mean_train_loss": train_loss})
                delete_dataset(train_dataset)

                # ========= eval for this epoch ==========
                policy = self.model
                if self.epoch % cfg.training.checkpoint_every == 0:
                    self.save_checkpoint(tag=f"checkpoint_{self.epoch}")
                    print("save checkpoint")
                # self.save_checkpoint()
                # print("save checkpoint")

                # # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        val_dataset = dp_baseline_dataset(cfg.task.val_dataset.zarr_path,
                                                  cfg.task.val_dataset.traj_num, cfg.task.val_dataset.train_mask_start, cfg.task.val_dataset.train_mask_end,
                                                  cfg.task.val_dataset.val_mask_start, cfg.task.val_dataset.val_mask_end,
                                                  pad_before=cfg.task.dataset.pad_before, pad_after=cfg.task.dataset.pad_after, horizon=cfg.task.dataset.horizon)
                        val_dataloader = DataLoader(val_dataset, **cfg.dataloader)
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss
                            wandb.log({"val_loss": val_loss})
                            if val_loss < min_val_loss:
                                min_val_loss = val_loss
                                # checkpointing
                                self.save_checkpoint()
                                print("save checkpoint")
                        delete_dataset(val_dataset)
                # ========= eval end for this epoch ==========
                policy.train()
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetHybridWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()