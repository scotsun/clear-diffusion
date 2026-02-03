import sys
from pathlib import Path
import argparse
import torch
import mlflow
import numpy as np

from mlflow.types import Schema, TensorSpec
from mlflow.models import ModelSignature


sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sd_vae.ae import VAE
from src.utils.exp_utils.train_utils import (
    load_cfg,
    build_first_stage_trainer,
    xavier_init,
)
from src.utils.exp_utils.visual import feature_swapping_plot
from src.utils.data_utils.camelyon import build_dataloader

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/camelyon.yaml")
    return parser.parse_args()


def main():
    args = get_args()
    cfg = load_cfg(args.config)

    dataloaders = build_dataloader(
        data_root="/hpc/group/engelhardlab/ms1008/image_data",
        batch_size=cfg[""],
        download=False,
    )

    # data signature
    img_size = cfg["data"]["img_size"]
    input_schema = Schema(
        [TensorSpec(np.dtype(np.float32), [-1, 1, img_size, img_size])]
    )
    output_schema = Schema(
        [TensorSpec(np.dtype(np.float32), [-1, 1, img_size, img_size])]
    )
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    # model
    vae = VAE(**cfg["vae"]).to(device)
    vae.apply(xavier_init)

    # trainer
    trainer = build_first_stage_trainer(
        cfg=cfg,
        trainer_class="CLEAR_VAEFirstStageTrainer",
        model=vae,
        signature=signature,
        device=device,
    )

    # train
    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment(args.exp_name)
    with mlflow.start_run() as run:
        mlflow.log_params(cfg["vae"] | cfg["trainer_param"])
        trainer.fit(
            epochs=cfg["train"]["epochs"],
            train_loader=dataloaders["train"],
            valid_loader=dataloaders["valid"],
        )

    # eval
    x = next(iter(dataloaders["test"]))["image"].to(device)
    best_model = mlflow.pytorch.load_model(f"runs:/{run.info.run_id}/best_model")
    with torch.no_grad():
        best_model.eval()
        _, posterior = best_model(x)
    z_c, z_s = posterior.mu.split_with_sizes(
        cfg["trainer_param"]["channel_split"], dim=1
    )
    select = torch.randint(0, 32, (5,)).tolist()
    feature_swapping_plot(
        z_c[select],
        z_s[select],
        x[select],
        best_model,
        img_size=96,
    )
