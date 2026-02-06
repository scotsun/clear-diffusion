import yaml
import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.distributed as dist

from datetime import timedelta

from src.trainers import EarlyStopping
from src.trainers.first_stage_trainer import CLEAR_VAEFirstStageTrainer
from src.trainers.cls_trainer import DownstreamMLPTrainer


def load_cfg(cfg_path) -> dict:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def setup_training():
    # check if in ddp env
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])

        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=600))

        return local_rank, world_size, rank, device, True
    else:
        local_rank = 0
        world_size = 1
        rank = 0

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(local_rank)

        return local_rank, world_size, rank, device, False


def build_first_stage_trainer(cfg, trainer_class, model, signature, device):
    match trainer_class:
        case "CLEAR_VAEFirstStageTrainer":
            trainer_class = CLEAR_VAEFirstStageTrainer
        case _:
            raise ValueError(f"Unknown trainer class: {trainer_class}")
    trainer = trainer_class(
        model=model,
        early_stopping=EarlyStopping(cfg["train"]["early_stopping_patience"]),
        verbose_period=2,
        device=device,
        model_signature=signature,
        args=cfg["trainer_param"],
    )
    return trainer


def build_cls_trainer(cfg, trainer_class, model, signature, device):
    match trainer_class:
        case "DownstreamMLPTrainer":
            trainer_class = DownstreamMLPTrainer
        case _:
            raise ValueError(f"Unknown trainer class: {trainer_class}")
    trainer = trainer_class(
        model=model,
        early_stopping=EarlyStopping(cfg["train"]["early_stopping_patience"]),
        verbose_period=2,
        device=device,
        model_signature=signature,
        args=cfg["trainer_param"],
    )
    return trainer


def xavier_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            init.ones_(m.weight)
            init.zeros_(m.bias)
