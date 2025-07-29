import os
import hydra
import torch
import random
import numpy as np
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig, OmegaConf
from loguru import logger
import shutil
from utils.misc import compute_model_dim
from utils.io import mkdir_if_not_exists
from utils.plot import Ploter
from datasets.base import create_dataset
from datasets.misc import collate_fn_general, collate_fn_squeeze_pcd_batch
from models.base import create_model

def load_ckpt(model: torch.nn.Module, ckpt_dir: str, save_model_separately: bool) -> (int, int):
    """ Load model and corresponding data

    Args:
        model: model to load the state dict
        ckpt_dir: directory where checkpoints are saved
        save_model_separately: flag indicating if checkpoints are saved separately

    Returns:
        epoch: last epoch
        step: last step
    """
    if save_model_separately:
        checkpoint_files = [f for f in os.listdir(ckpt_dir) if f.startswith('model_') and f.endswith('.pth')]
        if not checkpoint_files:
            return 0, 0

        latest_ckpt = max(checkpoint_files, key=lambda f: int(f.split('_')[1].replace('.pth', '')))
        ckpt_path = os.path.join(ckpt_dir, latest_ckpt)

    else:
        ckpt_path = os.path.join(ckpt_dir, 'model.pth')
        if not os.path.exists(ckpt_path):
            return 0, 0
    print("-" * 100)
    print("load model from ckpt")
    print("-"*100)
    logger.info(f'Loading checkpoint from {ckpt_path}')
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model'])
    epoch = checkpoint['epoch']
    step = checkpoint['step']
    return epoch, step

@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """ testing portal, test with multi gpus

    Args:
        cfg: configuration dict
    """
    ## set rank
    cfg.gpu = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(cfg.gpu)
    device = torch.device('cuda', cfg.gpu)
    torch.distributed.init_process_group(backend='nccl')

    ## compute modeling dimension according to task
    cfg.model.d_x = compute_model_dim(cfg.task)
    if os.environ.get('SLURM') is not None:
        cfg.slurm = True # update slurm config
        logger.remove(handler_id=0) # remove default handler

    ## set output logger and tensorboard
    ## Begin testing progress
    if cfg.gpu == 0:
        logger.add(cfg.exp_dir + '/test_runtime.log')  # change here 

        mkdir_if_not_exists(cfg.tb_dir)
        mkdir_if_not_exists(cfg.ckpt_dir)

        writer = SummaryWriter(log_dir=cfg.tb_dir)
        Ploter.setWriter(writer)

        logger.info('Configuration: \n' + OmegaConf.to_yaml(cfg))
        logger.info('Begin testing..')

    test_dataset = create_dataset(cfg.task.dataset, 'test', cfg.slurm)
    ## prepare dataset for test
    if cfg.gpu == 0:
        logger.info(f'Load test dataset size: {len(test_dataset)}')
    test_sampler = DistributedSampler(test_dataset,shuffle=True) # it uses to cooperate the dataparallel

    if cfg.model.scene_model.name == 'PointTransformer':
        collate_fn = collate_fn_squeeze_pcd_batch
    else:
        collate_fn = collate_fn_general
    
    test_dataloader = test_dataset.get_dataloader(
        sampler=test_sampler,
        batch_size=cfg.task.test.batch_size,
        collate_fn=collate_fn,
        num_workers=cfg.task.test.num_workers,
        pin_memory=True,
        drop_last = True
    )
    
    model = create_model(cfg, slurm=cfg.slurm, device=device)
    model.to(device=device)

    params = []
    nparams = []
    for n, p in model.named_parameters():
        # 'TODO: add more parameters to freeze'
        # if 'eps_model.out_layers.0' not in n and 'eps_model.out_layers.2' not in n:
        #     p.requires_grad = False
        if p.requires_grad:
            params.append(p)
            nparams.append(p.nelement())
            if cfg.gpu == 0:
                logger.info(f'add {n} {p.shape} for optimization')
    
    params_group = [
        {'params': params, 'lr': cfg.task.lr},
    ]
    if cfg.gpu == 0:
        logger.info(f'{len(params)} parameters for optimization.')
        logger.info(f'total model size is {sum(nparams)}.')

    ## convert to parallel
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[cfg.gpu], output_device=cfg.gpu, find_unused_parameters=True)
    
    # Resume from checkpoint if exists
    start_epoch, start_step = load_ckpt(model, cfg.ckpt_dir, cfg.save_model_seperately)

    step = start_step

    total_loss = 0
    outputs = 0

    for epoch in range(start_epoch, cfg.task.test.num_epochs):
        model.eval()
        if epoch > start_epoch:
            start_step = 0
        for it, data in enumerate(test_dataloader,start = (start_step % cfg.task.test.log_step)):
            for key in data:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(device)

            data['epoch'] = epoch
            outputs = model.cal_sample_loss(data,)
            
            ## plot loss only on first device
            if cfg.gpu == 0 and (step + 1) % cfg.task.test.log_step == 0:
                total_loss = outputs['loss'].item()
                log_str = f'[test] ==> Epoch: {epoch+1:3d} | Iter: {it+1:5d} | Step: {step+1:7d} | Loss: {total_loss:.3f}'
                logger.info(log_str)
                for key in outputs:
                    val = outputs[key].item() if torch.is_tensor(outputs[key]) else outputs[key]
                    Ploter.write({
                        f'test/{key}': {'plot': True, 'value': val, 'step': step},
                        # 'test/epoch': {'plot': True, 'value': epoch, 'step': step},
                    })

            step += 1

        # save epoch loss ----------------------
        if cfg.gpu == 0:
            for key in outputs:
                val = outputs[key].item() if torch.is_tensor(outputs[key]) else outputs[key]
                Ploter.write({
                    f'test_epoch/{key}': {'plot': True, 'value': val, 'step': step},
                    # 'test/epoch': {'plot': True, 'value': epoch, 'step': step},
                })
        # save epoch loss --------------------
            

    ## testing is over!
    if cfg.gpu == 0:
        writer.close() # close summarywriter and flush all data to disk
        logger.info('End testing..')

# def set_random_seed(seed = 2022):


if __name__ == '__main__':
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

