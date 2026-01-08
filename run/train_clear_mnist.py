import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.init as init
import mlflow
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torchvision.datasets import MNIST
from mlflow.types import Schema, TensorSpec
from mlflow.models import ModelSignature

from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).parent.parent))

from sd_vae.ae import VAE
from exp_utils.train_utils import load_cfg, build_first_stage_trainer
from exp_utils.visual import feature_swapping_plot
from data_utils.styled_mnist import corruptions
from data_utils.styled_mnist.data_utils import StyledMNISTGenerator, build_dataloaders

device = "cuda" if torch.cuda.is_available() else "cpu"


def xavier_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            init.ones_(m.weight)
            init.zeros_(m.bias)


def main():
    cfg = load_cfg("./config/mnist.yaml")
    # data
    np.random.seed(101)
    torch.manual_seed(101)
    mnist = MNIST("./data", train=True, download=False)
    generator = StyledMNISTGenerator(
        mnist,
        {
            lambda x: corruptions.rgb_change(x, "red"): 0.2,
            lambda x: corruptions.rgb_change(x, "green"): 0.2,
            lambda x: corruptions.rgb_change(x, "blue"): 0.2,
            lambda x: corruptions.rgb_change(x, "yellow"): 0.2,
            lambda x: corruptions.rgb_change(x, "magenta"): 0.2,
        },
    )
    dataloaders = build_dataloaders(generator)

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
    mlflow.set_experiment("test")
    with mlflow.start_run() as run:
        mlflow.log_params(cfg["trainer_param"])
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
    select = torch.randint(0, 128, (15,)).tolist()
    feature_swapping_plot(
        z_c[select],
        z_s[select],
        x[select],
        best_model,
        out_dir=cfg["out_dir"],
        run_id=run.info.run_id,
    )

    z_cs = []
    z_ss = []
    labels = []
    styles = []

    with torch.no_grad():
        best_model.eval()
        for batch in tqdm(dataloaders["test"]):
            x = batch["image"].to(device)
            _, posterior = best_model(x)
            z_c, z_s = posterior.sample().split_with_sizes(
                cfg["trainer_param"]["channel_split"], dim=1
            )
            z_cs.append(z_c.cpu())
            z_ss.append(z_s.cpu())
            labels.append(batch["label"])
            styles.append(batch["style"])

    z_cs = torch.cat(z_cs, dim=0)
    z_ss = torch.cat(z_ss, dim=0)
    labels = torch.cat(labels, dim=0)
    styles = torch.cat(styles, dim=0)

    tsne = TSNE(n_components=2, init="pca")
    z_2d = tsne.fit_transform(z_cs.view(z_cs.shape[0], -1).numpy())

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    axs0 = axs[0].scatter(
        z_2d[:, 0], z_2d[:, 1], c=labels.numpy(), cmap="tab10", alpha=0.2
    )
    _ = fig.colorbar(axs0, ax=axs[0])
    axs[0].set_title("color by content")

    style_labels = cfg["tsne"]["style_labels"]
    cmap = plt.get_cmap("Set1")
    colors = [cmap(i) for i in np.linspace(0, 1, len(style_labels))]
    for g in range(len(style_labels)):
        i = np.where(styles == g)[0]
        axs[1].scatter(
            z_2d[i, 0], z_2d[i, 1], alpha=0.2, c=colors[g], label=style_labels[g]
        )
    axs[1].legend()
    plt.savefig(
        os.path.join(cfg["out_dir"], run.info.run_id, "tsne_plot_1.png"),
        dpi=200,
        bbox_inches="tight",
    )

    X = z_ss.view(z_ss.shape[0], -1).cpu().numpy()
    y_content = labels.cpu().numpy()
    y_style = styles.cpu().numpy()

    N = X.shape[0]
    idx = np.random.choice(N, size=N, replace=False)

    X_sub = X[idx]
    content_sub = y_content[idx]
    style_sub = y_style[idx]

    tsne = TSNE(
        n_components=2,
        init="pca",
        perplexity=100,
        learning_rate=800,
        max_iter=8000,
        early_exaggeration=40,
        random_state=0,
    )
    z_2d = tsne.fit_transform(X_sub)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    axs0 = axs[0].scatter(
        z_2d[:, 0],
        z_2d[:, 1],
        c=content_sub,
        cmap="tab10",
        alpha=0.4,
        s=5,
    )
    _ = fig.colorbar(axs0, ax=axs[0])
    axs[0].set_title("color by content (5 types)")

    style_labels = cfg["tsne"]["style_labels"]
    cmap = plt.get_cmap("Set1")
    colors = [cmap(i) for i in np.linspace(0, 1, len(style_labels))]

    for g in range(len(style_labels)):
        i = np.where(style_sub == g)[0]
        axs[1].scatter(
            z_2d[i, 0],
            z_2d[i, 1],
            alpha=0.6,
            c=[colors[g]],
            s=5,
            label=style_labels[g],
        )

    axs[1].set_title("color by style")
    axs[1].legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(cfg["out_dir"], run.info.run_id, "tsne_plot_2.png"),
        dpi=200,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
