import os
import sys
from pathlib import Path
import argparse
import torch.nn as nn
import numpy as np
import mlflow

from torch.nn.parallel import DistributedDataParallel as DDP
from setproctitle import setproctitle
from dotenv import load_dotenv
from mlflow.types import Schema, TensorSpec
from mlflow.models import ModelSignature


sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sd_vae.ae import VAE
from src.utils.exp_utils.downstream_utils import get_flatten_dim
from src.utils.exp_utils.train_utils import (
    load_cfg,
    setup_training,
    build_cls_trainer,
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

    # data signature
    img_size = cfg["data"]["img_size"]
    n_class = cfg["data"]["n_class"]
    input_schema = Schema(
        [TensorSpec(np.dtype(np.float32), [-1, 1, img_size, img_size])]
    )
    output_schema = Schema([TensorSpec(np.dtype(np.float32), [-1, n_class])])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    # load model
    run_id, model_name = cfg["vae"]["run_id"], cfg["vae"]["model_name"]
    vae: VAE = mlflow.pytorch.load_model(
        f"runs:/{run_id}/{model_name}",
        map_location=device,
        dst_path="./tmp",
    )
    # cls_model
    cls_in_dim = get_flatten_dim(
        vae=vae,
        img_size=cfg["data"]["img_size"],
        channel=cfg["data"]["content_channel"],
    )
    cls_model = nn.Sequential(
        nn.Linear(cls_in_dim, cfg["cls_model"]["hidden_dim"]),
        nn.ReLU(),
        nn.Linear(cfg["cls_model"]["hidden_dim"], cfg["data"]["n_class"]),
    ).to(device)

    # trainer
    trainer = build_cls_trainer(
        cfg=cfg,
        trainer_class="DownstreamMLPTrainer",
        model=cls_model,
        vae=vae,
        signature=signature,
        device=device,
    )
    if is_distributed:
        trainer.model = DDP(trainer.model, device_ids=[rank])

    # train
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=RUN_NAME) as run:
        mlflow.log_params(cfg["cls_model"] | cfg["trainer_param"])
        trainer.fit(
            epochs=cfg["train"]["epochs"],
            train_loader=dataloaders["train"],
            valid_loader=dataloaders["valid"],
        )

    # test
    test_rlt = trainer.evaluate(dataloader=dataloaders["test"], verbose=True)
    print(test_rlt)


if __name__ == "__main__":
    main()
