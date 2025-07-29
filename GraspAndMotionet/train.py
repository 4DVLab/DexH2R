# import debugpy
# debugpy.listen(("localhost", 15000))
# debugpy.wait_for_client()  # 等待调试器连接

import os
import hydra
import torch
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from utils.io import mkdir_if_not_exists
from utils.plot import Ploter
from datasets.base import create_dataset
from datasets.misc import collate_fn_general
from models.base import create_model
from tqdm import tqdm
import colorama
import trimesh as tm
from os.path import join as pjoin


def save_ckpt(model: torch.nn.Module, epoch: int, step: int, path: str) -> None:
    """ Save current model and corresponding data

    Args:
        model: best model
        epoch: best epoch
        step: current step
        path: save path
    """
    
    saved_state_dict = {}
    model_state_dict = model.state_dict()
    for key in model_state_dict:
        saved_state_dict[key] = model_state_dict[key]
    
    logger.info('Saving model!!!' )
    
    # 1500
    checkpoint_files = sorted([f for f in os.listdir(os.path.dirname(path)) if f.startswith('model_') and f.endswith('.pth')],
                              key=lambda x: int(x.split('_')[1].split('.')[0]))
    if len(checkpoint_files) >= 10:
        os.remove(os.path.join(os.path.dirname(path), checkpoint_files[0]))
    try:
        print("try to save model")
        torch.save({
            'model': saved_state_dict,
            'epoch': epoch, 'step': step,
        }, path)
    except OSError as e:
        logger.error(f"Error saving model at {path}: {str(e)}")
    print("save model end")


def adapt_obs_weight_loading(model_ckpt,change_key_name,past_frame_num,global_offset):

    original_past_frame = 5
    hand_qpose_dim = 31
    hand_encode_part_offer = original_past_frame * hand_qpose_dim

    need_frame_begin_index = original_past_frame - past_frame_num
    hand_part = torch.arange(need_frame_begin_index * hand_qpose_dim,hand_encode_part_offer)

    pcd_dim = 1024
    pcd_encode_part_offset = hand_encode_part_offer + original_past_frame * pcd_dim
    pcd_part = torch.arange(hand_encode_part_offer + need_frame_begin_index * pcd_dim, pcd_encode_part_offset)
    
    original_ecode_length = 7300
    other_part = torch.arange(pcd_encode_part_offset,original_ecode_length)
    
    global_offset_part = torch.arange(global_offset)
    need_indices = torch.cat([hand_part, pcd_part,other_part], dim=-1) + global_offset
    need_indices_add_global_offset_part = torch.cat([global_offset_part,need_indices], dim=-1)
    for key_name in change_key_name:
        model_ckpt[key_name] = model_ckpt[key_name][...,need_indices_add_global_offset_part]
    return model_ckpt

def motion_adapt_weight(model_ckpt,future_frame,past_frames):
        print(colorama.Fore.RED +  "motion_net wright load from predict 10 frames , observation is 5")
        if future_frame != 10:
            print(colorama.Fore.RED +  "future_frame is not 10!!!")
            # process future predict frames num
            dec_pose_key_name = [key_name for key_name in model_ckpt if "dec_pose" in key_name]
            dec_trans_key_name = [key_name for key_name in model_ckpt if "dec_trans" in key_name]


            for key_name in dec_pose_key_name:
                model_ckpt[key_name] = model_ckpt[key_name][:28*future_frame]
            for key_name in dec_trans_key_name:
                model_ckpt[key_name] = model_ckpt[key_name][:3*future_frame]
        # preocess end

        # process past frames num
        if model_ckpt['module.motion_net_block.dec_bn1.weight'].shape[-1] == 7300 and past_frames != 5:
            print(colorama.Fore.RED +  "observation is not 5!!!")
            change_key_name = [
                                'module.motion_net_block.dec_bn1.weight',
                                'module.motion_net_block.dec_bn1.bias',
                                'module.motion_net_block.dec_bn1.running_mean',
                                'module.motion_net_block.dec_bn1.running_var',
                                'module.motion_net_block.dec_rb1.fc1.weight',
                                "module.motion_net_block.dec_rb1.fc3.weight"
                                ]
            model_ckpt = adapt_obs_weight_loading(model_ckpt,change_key_name,past_frames,0)
            change_key_name = [
                                'module.motion_net_block.dec_rb2.fc1.weight',
                                'module.motion_net_block.dec_rb2.fc3.weight'
                                ]
            model_ckpt = adapt_obs_weight_loading(model_ckpt,change_key_name,past_frames,2048) # motion net block n_neurons is 2048
            # process end
            model_ckpt = {k: v for k, v in model_ckpt.items() if 'dec_xyz' not in k and 'dec_dist' not in k}
        return model_ckpt


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
    logger.info(f'Loading checkpoint from {ckpt_path}')
    checkpoint = torch.load(ckpt_path)

    model_ckpt = checkpoint['model']

    # if cfg.model.name == "motion_net":
    #     model_ckpt = motion_adapt_weight(model_ckpt,cfg.task.dataset.future_frames,cfg.model.past_frames)


    model.load_state_dict(model_ckpt)



@hydra.main(version_base=None, config_path="./configs", config_name="default")# default,motion
def main(cfg: DictConfig) -> None:
    """ training portal, train with multi gpus

    Args:
        cfg: configuration dict
    """
    # if cfg.model.name == "motion_net":
    #     cfg.model.input_dim = cfg.model.past_frames * 31 + cfg.model.past_frames * 1024 + 1854


    device = 'cuda'

    logger.add(cfg.exp_dir + '/runtime.log')

    mkdir_if_not_exists(cfg.tb_dir)
    mkdir_if_not_exists(cfg.ckpt_dir)
    writer = SummaryWriter(log_dir=cfg.tb_dir)
    Ploter.setWriter(writer)

    logger.info('Configuration: \n' + OmegaConf.to_yaml(cfg))
    logger.info('Begin training..')

    train_dataset = create_dataset(cfg.task.dataset, 'train')
    ## prepare dataset for train
    logger.info(f'Load train dataset size: {len(train_dataset)}')

    
    train_dataloader = train_dataset.get_dataloader(
        batch_size=cfg.task.train.batch_size,
        collate_fn=collate_fn_general,
        num_workers=cfg.task.train.num_workers,
        pin_memory=True,
        drop_last = True,
        shuffle=True
    )
    
    ## create model and optimizer
    model = create_model(cfg, device=device)
    model.to(device=device)

    params = []
    nparams = []

    for n, p in model.named_parameters():
        # 'TODO: add more parameters to freeze'

        if p.requires_grad:
            params.append(p)
            nparams.append(p.nelement())
            if cfg.gpu == 0:
                logger.info(f'add {n} {p.shape} for optimization')
    
    params_group = [
        {'params': params, 'lr': cfg.task.lr},
    ]
    optimizer = torch.optim.Adam(params_group) # use adam optimizer in default
    if cfg.gpu == 0:
        logger.info(f'{len(params)} parameters for optimization.')
        logger.info(f'total model size is {sum(nparams)}.')

    if cfg.use_pretrain:
        load_ckpt(model, cfg.pretrain_model_dir_path,cfg.pretrain_model_index)


    total_loss = 0
    outputs = 0
    step = 0
    for epoch in range(0, cfg.task.train.num_epochs):
        model.train()

        for it, data in tqdm(enumerate(train_dataloader,start = 0),desc = "training one epoch", total = len(train_dataloader)):#(start_step % cfg.task.train.log_step)):
            for key in data:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(device)
            optimizer.zero_grad()
            data['epoch'] = epoch
            outputs = model(data)
            outputs['loss'].backward()
            optimizer.step()
            
            ## plot loss only on first device
            if  (step + 1) % cfg.task.train.log_step == 0:
                total_loss = outputs['loss'].item()
                log_str = f'[TRAIN] ==> Epoch: {epoch+1:3d} | Iter: {it+1:5d} | Step: {step+1:7d} | Loss: {total_loss:.3f}'
                logger.info(log_str)
                for key in outputs:
                    val = outputs[key].item() if torch.is_tensor(outputs[key]) else outputs[key]
                    Ploter.write({
                        f'train_iter/{key}': {'plot': True, 'value': val, 'step': step},
                    })

            step += 1

        # save epoch loss ----------------------

        for key in outputs.keys():
            val = outputs[key].item() if torch.is_tensor(outputs[key]) else outputs[key]
            Ploter.write({
                f'train_epoch/{key}': {'plot': True, 'value': val, 'step': epoch},

            })
        # save epoch loss --------------------

        ## save ckpt in epoch
        if (epoch + 1) % cfg.save_model_interval == 0:
            save_path = os.path.join(cfg.ckpt_dir, f'model_{epoch}.pth')
            save_ckpt(model=model, epoch=epoch, step=step, path=save_path)

    ## Training is over!
    writer.close() # close summarywriter and flush all data to disk
    logger.info('End training..')



if __name__ == '__main__':
    colorama.init(autoreset=True) 
    seed = 2022
    ## set random seed
    torch.backends.cudnn.benchmark = False     
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    main()

