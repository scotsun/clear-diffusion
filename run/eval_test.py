import sys
from pathlib import Path
import torch
import mlflow

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.exp_utils.train_utils import load_cfg
from src.utils.exp_utils.visual import feature_swapping_plot
from src.utils.data_utils.camelyon import build_dataloader

device = "cuda" if torch.cuda.is_available() else "cpu"

dataloaders = build_dataloader(
    data_root="/hpc/group/engelhardlab/ms1008/image_data",
    batch_size=32,
    download=False,
    num_workers=10,
)

run_id = "359962abd2384324889a0ae928ddf45e"
cfg = load_cfg("./config/camelyon.yaml")

# eval
x = next(iter(dataloaders["test"]))["image"].to(device)
best_model = mlflow.pytorch.load_model(
    f"runs:/{run_id}/best_model",
    map_location=device,
    dst_path="./tmp",
)
with torch.no_grad():
    best_model.eval()
    _, posterior = best_model(x)
z_c, z_s = posterior.mu.split_with_sizes(cfg["trainer_param"]["channel_split"], dim=1)
select = torch.randint(0, x.shape[0], (5,)).tolist()
feature_swapping_plot(
    z_c[select],
    z_s[select],
    x[select],
    best_model,
    img_size=cfg["data"]["img_size"],
)
