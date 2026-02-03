import os
import sys
from pathlib import Path
import argparse
import torch
import mlflow
import numpy as np

from dotenv import load_dotenv
from mlflow.types import Schema, TensorSpec
from mlflow.models import ModelSignature


sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sd_vae.ae import VAE
from src.utils.exp_utils.train_utils import (
    load_cfg,
    build_first_stage_trainer,
    xavier_init,
)
# from src.utils.data_utils.camelyon import build_dataloader


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

    print(os.getenv("DATA_ROOT"))


if __name__ == "__main__":
    main()
