import numpy as np
import yaml
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import mlflow
from mlflow.types import Schema, TensorSpec
from mlflow.models import ModelSignature

from sd_vae.ae import VAE
from trainers import EarlyStopping
from trainers.first_stage_trainer import CLEAR_VAEFirstStageTrainer

from modules.loss import SupCon, SNN, DenseSupCon, DenseSNN

import data_utils.styled_mnist.corruptions as corruptions
from data_utils.styled_mnist.data_utils import StyledMNISTGenerator, StyledMNIST



import torch.nn as nn
import torch.nn.init as init

import random

import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from exp_utils.visual import feature_swapping_plot
from datetime import datetime
import os
import math
from sklearn.manifold import TSNE
from functools import partial


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)

    elif isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

    elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.LayerNorm)):
        if m.weight is not None:
            init.ones_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)


def silu_xavier_init(m):
    SILU_GAIN = 1.4884  # √2 * √(π/8)
    
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=SILU_GAIN)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(m.weight, gain=SILU_GAIN)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)




def save_mu_channels(mu, start, end, title_prefix, filename, nrow_grid=16, out_dir="outputs/images"):
    """
    mu: Tensor (B, C, H, W)
    start/end: channel slice [start, end)
    """
    os.makedirs(out_dir, exist_ok=True)

    C = end - start
    n_cols = int(math.ceil(math.sqrt(C)))
    n_rows = int(math.ceil(C / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten() if C > 1 else [axes]

    for k, i in enumerate(range(start, end)):
        grid = make_grid(
            mu[:, i][:, None, :, :],   # (B,1,H,W)
            nrow=nrow_grid,
            normalize=True
        ).cpu().permute(1, 2, 0)

        axes[k].imshow(grid)
        axes[k].set_title(f"{title_prefix}[{i}]")
        axes[k].axis("off")

    for j in range(C, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")



def build_corruption_dict(corruption_configs):
    """    
    Args:
        corruption_configs: list of dict, each with:
            - name: corruption function name
            - params: dict of parameters
            - probability: float
    
    Returns:
        dict: {corruption_function: probability}
    """
    corruption_dict = {}
    
    for config in corruption_configs:
        corruption_name = config['name']
        params = config.get('params', {})
        probability = config['probability']
        
        if not hasattr(corruptions, corruption_name):
            raise ValueError(f"Unknown corruption: {corruption_name}")
        
        corruption_func = getattr(corruptions, corruption_name)
        
        if params:
            wrapped_func = partial(corruption_func, **params)
        else:
            wrapped_func = corruption_func
        
        corruption_dict[wrapped_func] = probability
    
    return corruption_dict


def load_styled_mnist_from_config(config):
    dataset_config = config['dataset']
    
    mnist = MNIST(
        root=dataset_config['root'],
        train=True,
        download=True
    )
    
    generator_config = dataset_config.get('generator', {})
    corruption_configs = generator_config.get('corruptions', [])
    
    if not corruption_configs:
        raise ValueError("No corruptions specified in config")
    
    corruption_dict = build_corruption_dict(corruption_configs)
    
    generator = StyledMNISTGenerator(mnist, corruption_dict)

    styled_mnist = StyledMNIST(
        generator,
        transforms.Compose([
            transforms.ToTensor(),
            lambda img: img / 255.0,
        ])
    )


    return styled_mnist

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config.get('seed', 42))
    dataset_config = config['dataset']
    data_name = dataset_config['name']
    input_shape = dataset_config.get('input_shape', [3, 32, 32])
    if data_name in ['mnist', 'styled_mnist']:
        dataset = load_styled_mnist_from_config(config)
        
    
    # to do: add ways to analyze data
    split_sizes = config['dataset']['split_sizes']
    train, test, valid = random_split(dataset, split_sizes)
    train_loader = DataLoader(train, **dataset_config['train_loader'])
    valid_loader = DataLoader(valid, **dataset_config['valid_loader'])
    test_loader = DataLoader(test, **dataset_config['test_loader'])

    params = config['hyperparameters']
    vae_config = config['vae_architecture']
    contrastive_config = config['contrastive_module']
    device = torch.device(config['device'])

    input_schema = Schema([TensorSpec(np.dtype(np.float32), 
                                    [-1] + config['data']['input_shape'])])
    output_schema = Schema([TensorSpec(np.dtype(np.float32), 
                                    [-1] + config['data']['output_shape'])])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    vae = VAE(
        channels=vae_config['channels'],
        channel_multipliers=vae_config['channel_multipliers'],
        n_resnet_blocks=vae_config['n_resnet_blocks'],
        x_channels=vae_config['x_channels'],
        z_channels=vae_config['z_channels'],
        norm_channels=vae_config['norm_channels'],
        n_heads=vae_config['n_heads'],
    ).to(device)

    if config['trainer']['weight_init'] == 'silu_xavier_init':
        vae.apply(silu_xavier_init)

    trainer = CLEAR_VAEFirstStageTrainer(
        model=vae,
        early_stopping=EarlyStopping(
            patience=config['trainer']['early_stopping']['patience']
        ),
        verbose_period=config['trainer']['verbose_period'],
        device=device,
        model_signature=signature,
        args={
            "beta": params['beta'],
            "gamma_1": params['gamma_1'],
            "gamma_2": params['gamma_2'],
            "vae_lr": params['lr'],
            "channel_split": vae_config['channel_split'],
            "contrastive_module": contrastive_config,
        },
    )


    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment("test")
    with mlflow.start_run():
        mlflow.log_params(params)
        trainer.fit(epochs=config['trainer']['num_epochs'], train_loader=train_loader, valid_loader=valid_loader)


    print("content tau:", trainer.contrastive_criterions['global']['content'].log_tau.exp().item())
    print("style tau:", trainer.contrastive_criterions['global']['style'].log_tau.exp().item())
    x = next(iter(test_loader))['image'].to(device)
    best_model = trainer.model
    
    with torch.no_grad():
        best_model.eval()
        xhat, posterior = best_model(x)

    


    mu = posterior.mu
    content_dim, style_dim = config["vae_architecture"]["channel_split"]  # [3, 12]
    C_total = mu.shape[1]
    assert content_dim + style_dim == C_total, f"split sum {content_dim+style_dim} != mu channels {C_total}"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # content: [0, content_dim)
    save_mu_channels(
        mu,
        start=0,
        end=content_dim,
        title_prefix="content_mu",
        filename=f"{timestamp}_content_mu.png",
        out_dir= config['image_path']
    )

    # style: [content_dim, content_dim+style_dim)
    save_mu_channels(
        mu,
        start=content_dim,
        end=content_dim + style_dim,
        title_prefix="style_mu",
        filename=f"{timestamp}_style_mu.png",
        out_dir= config['image_path']
    )
    z_c, z_s = mu.split_with_sizes(vae_config['channel_split'], dim=1)
    x = next(iter(test_loader))['image'].to(device)

    select = torch.randint(0, len(x), (min(16, len(x)),)).tolist()
    feature_swapping_plot(
        z_c[select],
        z_s[select],
        x[select],
        best_model,
        save=True,
        name="swap_content_style",
        out_dir= config['image_path']
    )

    z_cs = []
    z_ss = []
    labels = []
    styles = []

    with torch.no_grad():
        best_model.eval()
        for batch in tqdm(test_loader):
            x = batch['image'].to(device)
            _, posterior = best_model(x)
            z_c, z_s = posterior.sample().split_with_sizes(vae_config['channel_split'], dim=1)
            z_cs.append(z_c.cpu())
            z_ss.append(z_s.cpu())
            labels.append(batch['label'])
            styles.append(batch['style'])

    z_cs = torch.cat(z_cs, dim=0)
    z_ss = torch.cat(z_ss, dim=0)
    labels = torch.cat(labels, dim=0)
    styles = torch.cat(styles, dim=0)

    tsne = TSNE(n_components=2, init='pca')
    z_2d = tsne.fit_transform(z_cs.view(z_cs.shape[0], -1).numpy())

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    axs0 = axs[0].scatter(z_2d[:, 0], z_2d[:, 1], c=labels.numpy(), cmap='tab10', alpha=0.2)
    cbar = fig.colorbar(axs0, ax=axs[0])
    axs[0].set_title('color by content')

    style_labels = config['tsne']['style_labels']
    cmap = plt.get_cmap('Set1')
    colors = [cmap(i) for i in np.linspace(0, 1, len(style_labels))]
    for g in range(len(style_labels)):
        i = np.where(styles == g)[0]
        axs[1].scatter(z_2d[i,0], z_2d[i,1], alpha=0.2, c=colors[g], label=style_labels[g])
    axs[1].legend()
    plt.savefig(os.path.join(config['image_path'], "tsne_plot_1.png"), dpi=200, bbox_inches="tight")


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
        init='pca',
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
        cmap='tab10',
        alpha=0.4,
        s=5,
    )
    cbar = fig.colorbar(axs0, ax=axs[0])
    axs[0].set_title('color by content (5 types)')

    style_labels = config['tsne']['style_labels']
    cmap = plt.get_cmap('Set1')
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

    axs[1].set_title('color by style')
    axs[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config['image_path'], "tsne_plot_2.png"), dpi=200, bbox_inches="tight")




if __name__ == "__main__":
    main()

