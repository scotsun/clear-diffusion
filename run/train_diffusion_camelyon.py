import mlflow
import torch
import torch.nn as nn
import yaml
import torch
import mlflow

from src.sd_vae.unet_base import UNetConditional
from src.scheduler.linear_noise_scheduler import LinearNoiseScheduler
from src.trainers.ldm import LDMTrainer
from src.utils.data_utils.camelyon import build_dataloader

def load_vae_from_mlflow(
    run_id      : str,
    artifact    : str = "best_model",
    tracking_uri: str = "mlruns",
    device      : torch.device = torch.device("cpu"),
) -> nn.Module:
    """
    Load a pretrained VAE from a local MLflow run and freeze all parameters.

    Args:
        run_id       : MLflow run ID string
        artifact     : artifact name used in mlflow.pytorch.log_model()
        tracking_uri : local mlruns directory (relative to project root)
        device       : target device

    Returns:
        vae: frozen nn.Module ready for use as encoder/decoder
    """
    mlflow.set_tracking_uri(tracking_uri)
    vae = mlflow.pytorch.load_model(f"runs:/{run_id}/{artifact}")
    vae = vae.to(device).eval()

    for p in vae.parameters():
        p.requires_grad_(False)

    return vae


def main():
    # ── Load config ───────────────────────────────────────────────────────────────
    with open("config/ldm.yaml") as f:
        cfg = yaml.safe_load(f)
    print("Config loaded:" + "\n" + yaml.dump(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ── Load frozen VAE via run_id ────────────────────────────────────────────────
    vae = load_vae_from_mlflow(
        run_id       = cfg["vae"]["run_id"],
        artifact     = cfg["vae"]["artifact"],
        tracking_uri = cfg["vae"]["tracking_uri"],
        device       = device,
    )
    
    # ── Build UNet denoiser ───────────────────────────────────────────────────────
    unet = UNetConditional(
        im_channels  = cfg["model_config"]["im_channels"],
        model_config = cfg["model_config"],
    ).to(device)
    
    # ── Build noise scheduler ─────────────────────────────────────────────────────
    scheduler = LinearNoiseScheduler(**cfg["scheduler_param"])
    
    # ── Build LDMTrainer ──────────────────────────────────────────────────────────
    trainer = LDMTrainer(
        model          = unet,
        vae            = vae,
        scheduler      = scheduler,
        early_stopping = None,
        verbose_period = cfg["trainer_param"]["verbose_period"],
        device         = device,
        model_signature= None,
        args           = cfg["trainer_param"],
    )
    
    
    
    # ── Build dataloaders ─────────────────────────────────────────────────────────
    dataloaders = build_dataloader(
        data_root     = cfg["data"]["data_root"],
        batch_size    = 256,
        download      = False,
        num_workers   = 32,
        is_distributed= False,
    )
    
    # dataloaders must have keys "train" and "val"
    # each batch: {"image": (B,C,H,W), "label": (B,)}
    
    # ── MLflow experiment ─────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(cfg["vae"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    
    with mlflow.start_run(run_name=cfg["mlflow"]["run_name"]) as run:
        print(f"MLflow Run ID: {run.info.run_id}")
        mlflow.log_params({
            "num_timesteps" : cfg["scheduler_param"]["num_timesteps"],
            "lr"            : cfg["trainer_param"]["lr"],
            "p_uncond"      : cfg["trainer_param"]["p_uncond"],
            "guidance_scale": cfg["sample_param"]["guidance_scale"],
            "vae_run_id"    : cfg["vae"]["run_id"],
        })
    
        best_val_loss = float("inf")
    
        for epoch in range(cfg["train"]["epochs"]):
            verbose = (epoch % cfg["trainer_param"]["verbose_period"] == 0)
    
            # ── Train ─────────────────────────────────────────────────
            trainer._train(dataloaders["train"], verbose=verbose, epoch_id=epoch)
    
            # ── Validate ──────────────────────────────────────────────
            result   = trainer._valid(dataloaders["valid"], verbose=verbose, epoch_id=epoch)
            val_loss = result["callback_metric"]
            mlflow.log_metrics(result["logged_metrics"], step=epoch)
    
            # ── Checkpoint best model ──────────────────────────────────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                mlflow.pytorch.log_model(unet, artifact_path="best_model")
                print(f"  → saved best model (val_loss={val_loss:.4f})")
    
            # ── Early stopping ────────────────────────────────────────
            # add your EarlyStopping logic here if needed
    
        # ── Final sample for visual check ─────────────────────────────
        class_ids = torch.arange(cfg["trainer_param"]["num_classes"]).to(device)
        samples   = trainer.sample(
            n_samples      = cfg["sample_param"]["n_samples"],
            class_ids      = class_ids.repeat(
                                 cfg["sample_param"]["n_samples"] //
                                 cfg["trainer_param"]["num_classes"] + 1
                             )[:cfg["sample_param"]["n_samples"]],
            latent_shape   = tuple(cfg["sample_param"]["latent_shape"]),
            guidance_scale = cfg["sample_param"]["guidance_scale"],
            show_progress  = True,
        )
        # samples: (n_samples, C, H, W) — save or log as needed
        print(f"Sampled output shape: {samples.shape}")


if __name__ == "__main__":
    main()
