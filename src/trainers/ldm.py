# src/trainers/ldm.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import mlflow

from tqdm import tqdm
from torch.utils.data import DataLoader
from mlflow.models import ModelSignature

from . import EarlyStopping, Trainer
from .ddpm import DDPMTrainer


class LDMTrainer(DDPMTrainer):
    """
    Latent Diffusion Model trainer on top of CLEAR VAE.

    Pipeline (train):
        x  ->  vae.encoder  ->  posterior.mu  ->  z_c  (content latent)
        z_c  ->  DDPM forward noise  ->  z_t
        UNetConditional(z_t, t, class_onehot)  ->  eps_pred
        DSM loss on eps_pred vs eps

    Pipeline (sample):
        z_T ~ N(0,I)  ->  DDPM reverse  ->  z_c_pred
        z_c_pred  ->  vae.decoder  ->  xhat

    Expected extra args keys:
        num_classes  : int
        channel_split: list[int]   e.g. [2, 2]
        p_uncond     : float       CFG dropout prob (default 0.1)
    """

    LATENT_SCALE = 0.18215   # matches VAE forward: z *= 0.18215

    def __init__(
        self,
        model          : nn.Module,        # UNetConditional
        vae            : nn.Module,        # pretrained CLEAR VAE
        scheduler,
        early_stopping : EarlyStopping | None,
        verbose_period : int,
        device         : torch.device,
        model_signature: ModelSignature,
        args           : dict,
    ) -> None:
        super().__init__(
            model, scheduler, early_stopping,
            verbose_period, device, model_signature, args,
        )
        self.vae          = vae.to(device).eval()
        self.num_classes  = args["num_classes"]
        self.channel_split= args["channel_split"]       # e.g. [2, 2]
        self.p_uncond     = args.get("p_uncond", 0.1)

        # freeze VAE entirely
        for p in self.vae.parameters():
            p.requires_grad_(False)

    # ── DDPMTrainer hooks ──────────────────────────────────────────────────────

    def _get_input(self, batch: dict) -> torch.Tensor:
        """
        Encode image -> z_c (content latent), scaled by 0.18215.
        Uses posterior.mu (deterministic, no noise) for stable diffusion training.
        """
        x = batch["image"].to(self.device)
        with torch.no_grad():
            moments  = self.vae.encoder(x)
            from src.modules.distribution import IsotropicNormalDistribution
            posterior = IsotropicNormalDistribution(moments)
            z_c, _   = posterior.mu.split_with_sizes(self.channel_split, dim=1)
            z_c      = z_c * self.LATENT_SCALE                    # (B, C_lat, H, W)
        return z_c


        
    def _get_cond(self, batch: dict) -> dict:
        """
        One-hot class conditioning with CFG dropout during training.
        Null condition = all-zeros one-hot (matches UNetConditional's null_class_emb).
        """
        labels = batch["label"].to(self.device)                   # (B,)
        B      = labels.shape[0]

        one_hot = torch.zeros(B, self.num_classes, device=self.device)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)             # (B, num_classes)

        # CFG dropout: null out p_uncond fraction during training
        if self.model.training and self.p_uncond > 0:
            drop_mask        = torch.rand(B, device=self.device) < self.p_uncond
            one_hot[drop_mask] = 0.0                              # zeros = unconditional

        return {"class": one_hot}

    # ── Sampling ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        n_samples      : int,
        class_ids      : torch.Tensor,
        latent_shape   : tuple = (4, 12, 12),
        guidance_scale : float = 7.5,
        show_progress  : bool  = True,
        ref_images     : torch.Tensor = None,
    ) -> torch.Tensor:
    
        one_hot = torch.zeros(n_samples, self.num_classes, device=self.device)
        one_hot.scatter_(1, class_ids.to(self.device).unsqueeze(1), 1.0)
        cond_input = {"class": one_hot}
    
        # reverse diffusion -> z_c (still in scaled space)
        z_c_scaled = super().sample(
            n_samples      = n_samples,
            sample_shape   = latent_shape,
            cond_input     = cond_input,
            guidance_scale = guidance_scale,
            show_progress  = show_progress,
        )
    
        # unscale z_c back to VAE decoder's expected range
        z_c = z_c_scaled / self.LATENT_SCALE
    
        # get z_s from reference images or use zeros
        if ref_images is not None:
            ref_images = ref_images.to(self.device)
            from src.modules.distribution import IsotropicNormalDistribution
            moments   = self.vae.encoder(ref_images)
            posterior = IsotropicNormalDistribution(moments)
            _, z_s    = posterior.mu.split_with_sizes(self.channel_split, dim=1)
            # z_s is already in unscaled space (no LATENT_SCALE applied)
        else:
            z_s_channels = sum(self.channel_split) - self.channel_split[0]
            z_s = torch.zeros(
                n_samples, z_s_channels, *z_c.shape[2:],
                device=self.device
            )
    
        z_full = torch.cat([z_c, z_s], dim=1)        # (B, C_lat_full, H, W)
        return self.vae.decoder(z_full)
    


    @torch.no_grad()
    def _ddpm_reverse(
        self,
        z_t           : torch.Tensor,
        cond_input    : dict,
        guidance_scale: float,
        latent_shape  : tuple,
        show_progress : bool = False,
    ) -> torch.Tensor:
        """Run reverse diffusion from a given z_t."""
        from tqdm import tqdm
        timesteps = list(reversed(range(self.T)))
        if show_progress:
            timesteps = tqdm(timesteps, desc="Reverse diffusion")
    
        for t in timesteps:
            t_batch  = torch.full((z_t.shape[0],), t, device=self.device, dtype=torch.long)
            use_cfg  = (cond_input is not None) and (guidance_scale > 1.0)
    
            if not use_cfg:
                noise_pred = self.model(z_t, t_batch, cond_input=cond_input)
            else:
                eps_cond   = self.model(z_t, t_batch, cond_input=cond_input)
                eps_uncond = self.model(z_t, t_batch,
                                       cond_input={k: torch.zeros_like(v)
                                                   for k, v in cond_input.items()})
                noise_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
    
            z_t, _ = self.scheduler.sample_prev_timestep(z_t, noise_pred, t)
    
        return z_t.clamp(-1.0, 1.0)
