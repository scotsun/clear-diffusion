import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import mlflow

from tqdm import tqdm
from torch.utils.data import DataLoader
from mlflow.models import ModelSignature

import torch.distributed as dist
from . import EarlyStopping, Trainer




def is_main_process() -> bool:
    """Returns True if not using distributed training, or if rank == 0."""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0

    
class DDPMTrainer(Trainer):
    """
    DDPM trainer using Denoising Score Matching loss.

    DSM loss: L = E[|| eps - eps_theta(x_t, t) ||^2]

    Score estimate: s_theta(x_t, t) = -eps_theta(x_t, t) / sqrt(1 - ᾱ_t)
    """

    def __init__(
        self,
        model         : nn.Module,
        scheduler,
        early_stopping: EarlyStopping | None,
        verbose_period: int,
        device        : torch.device,
        model_signature: ModelSignature,
        args          : dict,
    ) -> None:
        super().__init__(
            model,
            early_stopping,
            verbose_period,
            device,
            model_signature,
            args,
        )
        self.scheduler = scheduler
        self.T         = scheduler.num_timesteps
        self.opt       = optim.AdamW(model.parameters(), lr=args["lr"])

    # ── Score Matching  ──────────────────────────────────

    def _dsm_loss(
        self,
        x0        : torch.Tensor,
        t         : torch.Tensor,
        cond_input: dict | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Denoising Score Matching loss.

        L_DSM = E[|| eps - eps_theta(x_t, t) ||^2]

        Score recovered as:
            s_theta(x_t, t) = -eps_theta / sqrt(1 - ᾱ_t)
        """
        noise = torch.randn_like(x0)
        x_t   = self.scheduler.add_noise(x0, noise, t)

        noise_pred = (
            self.model(x_t, t, cond_input=cond_input)
            if cond_input is not None
            else self.model(x_t, t)
        )

        loss = F.mse_loss(noise_pred, noise)

        # score estimate for monitoring
        sqrt_one_minus_ab = (
    self.scheduler.sqrt_one_minus_alpha_cum_prod.to(x0.device)[t][:, None, None, None]
)
        score = -noise_pred / (sqrt_one_minus_ab + 1e-8)
        score_norm = score.flatten(1).norm(dim=1).mean()

        return loss, {"dsm_loss": loss.item(), "score_norm": score_norm.item()}

    # ── Hooks for subclasses ──────────────────────────────────────

    def _get_input(self, batch: dict) -> torch.Tensor:
        """Returns tensor to denoise. Override in LDM to return latent."""
        return batch["image"].to(self.device)

    def _get_cond(self, batch: dict) -> dict | None:
        """Returns cond_input. Override for conditional models."""
        return None

    # ── Train / Eval ──────────────────────────────────────────────

    def _train(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
        self.model.train()
        device = self.device

        with tqdm(dataloader, unit="batch", mininterval=0, disable=not verbose) as bar:
            bar.set_description(f"Epoch {epoch_id}")
            for batch_id, batch in enumerate(bar):
                x0 = self._get_input(batch)
                t  = torch.randint(0, self.T, (x0.shape[0],), device=device)

                loss, metrics = self._dsm_loss(x0, t, self._get_cond(batch))

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                bar.set_postfix(**metrics)
                cur_step = epoch_id * len(dataloader) + batch_id
                if cur_step % 50 == 0 and is_main_process():
                    mlflow.log_metrics(metrics, step=cur_step)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, verbose: bool):
        self.model.eval()
        total_loss = 0.0

        with tqdm(dataloader, unit="batch", mininterval=0, disable=not verbose) as bar:
            for batch in bar:
                x0 = self._get_input(batch)
                t  = torch.randint(0, self.T, (x0.shape[0],), device=self.device)
                loss, _ = self._dsm_loss(x0, t, self._get_cond(batch))
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def _valid(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
        val_loss = self.evaluate(dataloader, verbose)
        if verbose:
            print(f"epoch {epoch_id} | val_loss: {val_loss:.6f}")
        return {
            "callback_metric": val_loss,
            "logged_metrics"  : {"val_loss": val_loss},
        }

    # ── Sampling ──────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        n_samples     : int,
        sample_shape  : tuple,
        cond_input    : dict  = None,
        guidance_scale: float = 1.0,
        show_progress : bool  = True,
    ) -> torch.Tensor:
        self.model.eval()
        device  = self.device
        C, H, W = sample_shape

        x_t = torch.randn(n_samples, C, H, W, device=device)

        timesteps = list(reversed(range(self.T)))
        if show_progress:
            timesteps = tqdm(timesteps, desc="DDPM Sampling")

        for t in timesteps:
            t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
            use_cfg = (cond_input is not None) and (guidance_scale > 1.0)
            if not use_cfg:
                noise_pred = (
                    self.model(x_t, t_batch, cond_input=cond_input)
                    if cond_input else self.model(x_t, t_batch)
                )
            else:
                eps_cond   = self.model(x_t, t_batch, cond_input=cond_input)
                eps_uncond = self.model(x_t, t_batch,
                                        cond_input={k: torch.zeros_like(v)
                                                    for k, v in cond_input.items()})
                noise_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            x_t, _ = self.scheduler.sample_prev_timestep(x_t, noise_pred, t)

        return x_t.clamp(-1.0, 1.0)
