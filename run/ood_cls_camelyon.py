import os
import sys
from pathlib import Path
import argparse
import torch
import mlflow

from setproctitle import setproctitle
from dotenv import load_dotenv
from mlflow.types import Schema, TensorSpec
from mlflow.models import ModelSignature


sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sd_vae.ae import VAE
from src.utils.exp_utils.train_utils import (
    load_cfg,
    setup_training,
    build_first_stage_trainer,
    xavier_init,
)
from src.utils.data_utils.camelyon import build_dataloader


def get_args():
    parser = argparse.ArgumentParser(description="OOD CLS on Camelyon")
    parser.add_argument("--config", type=str, default="./config/ood_cls_camelyon.yaml")
    parser.add_argument("--backend-uri", type=str, default="./mlruns")
    parser.add_argument("--experiment-name", type=str)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    load_dotenv()
    args = get_args()
    cfg = load_cfg(args.config)
    MLFLOW_URI = args.backend_uri
    EXPERIMENT_NAME = args.experiment_name
    RUN_NAME = args.run_name

    # set multi-gpu
    os.environ["OMP_NUM_THREADS"] = "1"
    local_rank, world_size, rank, device, is_distributed = setup_training()
    setproctitle(f"ood-cls-camelyon-{local_rank}")
    print(f"local_rank: {local_rank}")
    print(f"process {rank}/{world_size} using device {device}\n")

    # data loading
    dataloaders = build_dataloader(
        data_root=os.getenv("CAMELYON_DATA_ROOT"),
        batch_size=cfg["data"]["batch_size"],
        download=False,
        num_workers=10,
        is_distributed=is_distributed,
    )

    # load model
    run_id, model_name = cfg["vae"]["run_id"], cfg["vae"]["model_name"]
    vae = mlflow.pytorch.load_model(
        f"runs:/{run_id}/{model_name}",
        map_location=device,
        dst_path="./tmp",
    )
    # cls_model

    # TODO: ddp both modules


if __name__ == "__main__":
    main()
