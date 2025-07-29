from typing import Dict, List
import torch.nn as nn
from omegaconf import DictConfig
from utils.registry import Registry


MODEL = Registry('Model')
DIFFUSER = Registry('Diffuser')


def create_model(cfg: DictConfig, *args: List, **kwargs: Dict) -> nn.Module:


    return MODEL.get(cfg.model.name)(cfg.model, *args, **kwargs)
