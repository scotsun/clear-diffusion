import torch
import torch.nn as nn
import numpy as np

from src.sd_vae.ae import VAE
from src.trainers.cls_trainer import DownstreamMLPTrainer


def get_flatten_dim(vae: VAE, img_size: int, channel: int):
    enc = vae.encoder
    n_downsample = enc.n_resolutions - 1
    h_size = img_size // (2**n_downsample)
    return channel * h_size * h_size


def evaluate_loaded_vae(
    vae: VAE, train_loader, valid_loader, test_loader, device, n_class=2
):
    vae.to(device)
    vae.eval()

    for p in vae.parameters():
        p.requires_grad = False
    print("Calculating MLP input dimension...")

    try:
        batch = next(iter(train_loader))

        if isinstance(batch, dict):
            if "x" in batch:
                x_dummy = batch["x"]
            elif "image" in batch:
                x_dummy = batch["image"]
            elif "data" in batch:
                x_dummy = batch["data"]
            else:
                x_dummy = list(batch.values())[0]
        else:
            x_dummy = batch[0]

        x_dummy = x_dummy[0:1].to(device)

    except Exception as e:
        print(f"Error getting dummy input: {e}. Fallback to random input (3x96x96).")
        x_dummy = torch.randn(1, 3, 96, 96).to(device)

    with torch.no_grad():
        moments = vae.encoder(x_dummy)

        c = moments.shape[1] // 2
        h, w = moments.shape[2], moments.shape[3]

        flatten_dim = c * h * w

    print(f"Detected encoder output shape: {moments.shape}")
    print(f"MLP input dimension (flattened): {flatten_dim}")

    mlp = nn.Sequential(
        nn.Linear(flatten_dim, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, n_class),
    ).to(device)

    optimizer = torch.optim.Adam(mlp.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    trainer = DownstreamMLPTrainer(vae, mlp, optimizer, criterion, 1, device)

    print("Training downstream MLP classifier...")
    trainer.fit(1, train_loader, valid_loader)

    print("Evaluating on test set...")
    (aupr_scores, auroc_scores), acc = trainer.evaluate(test_loader, False, 0)

    results = {
        "acc": round(float(acc), 3),
        "pr": {
            "overall": round(np.mean(list(aupr_scores.values())), 3),
            "stratified": aupr_scores,
        },
        "roc": {
            "overall": round(np.mean(list(auroc_scores.values())), 3),
            "stratified": auroc_scores,
        },
    }

    return results
