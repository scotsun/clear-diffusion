import os
import sys
from pathlib import Path
import argparse
import torch
import mlflow
import torch.distributed as dist
import numpy as np

from torch.nn.parallel import DistributedDataParallel as DDP
from setproctitle import setproctitle
from dotenv import load_dotenv
from mlflow.types import Schema, TensorSpec
from mlflow.models import ModelSignature

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sd_vae.ae import VAE
from src.utils.exp_utils.train_utils import (
    load_cfg,
    build_first_stage_trainer,
    setup_training,
    xavier_init,
)
from src.utils.exp_utils.visual import feature_swapping_plot
from src.utils.data_utils.camelyon import build_dataloader

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/camelyon.yaml")
    parser.add_argument("--backend-uri", type=str, default="./mlruns")
    parser.add_argument("--experiment-name", type=str)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


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
    setproctitle(f"train-clear-camelyon-{local_rank}")
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
    if is_distributed:
        trainer.model = DDP(trainer.model, device_ids=[local_rank])

    # train
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=RUN_NAME) as run:
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
    select = torch.randint(0, x.shape[0], (5,)).tolist()
    feature_swapping_plot(
        z_c[select],
        z_s[select],
        x[select],
        best_model,
        img_size=img_size,
    )

    # end
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
