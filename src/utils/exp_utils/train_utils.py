import yaml

from src.trainers import EarlyStopping
from src.trainers.first_stage_trainer import CLEAR_VAEFirstStageTrainer


def load_cfg(cfg_path) -> dict:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


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
