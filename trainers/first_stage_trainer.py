import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from sd_vae.ae import VAE
from . import EarlyStopping, Trainer


class VAEFirstStageTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        early_stopping: EarlyStopping | None,
        verbose_period: int,
        device: torch.device,
        transform=None,
    ) -> None:
        super().__init__(
            model, optimizer, early_stopping, verbose_period, device, transform
        )

    def _train(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
        vae: VAE = self.model
        vae.train()
        opt = self.optimizer
        device = self.device

        with tqdm(dataloader, unit="batch", mininterval=0, disable=not verbose) as bar:
            bar.set_description(f"Epoch {epoch_id}")
            for batch in bar:
                x = batch["image"].to(device)
                if self.transform:
                    x = self.transform(x)
                opt.zero_grad()
                xhat, posterior = vae(x)
                rec_loss = F.mse_loss(xhat, x, reduction="none").sum(dim=(1, 2, 3))
                kl_loss = posterior.kl()

                loss = rec_loss.mean() + kl_loss.mean()
                loss.backward()
                opt.step()

                bar.set_postfix(rec_loss=rec_loss.item(), kl_loss=kl_loss.item())
        return

    def evaluate(self, dataloader: DataLoader, verbose: bool):
        vae: VAE = self.model
        vae.eval()
        device = self.device

        total_rec_loss, total_kl_loss = 0.0, 0.0
        with torch.no_grad():
            for batch in tqdm(
                dataloader, unit="batch", mininterval=0, disable=not verbose
            ):
                x = batch["image"].to(device)
                if self.transform:
                    x = self.transform(x)
                xhat, posterior = vae(x)
                rec_loss = F.mse_loss(xhat, x, reduction="none").sum(dim=(1, 2, 3))
                kl_loss = posterior.kl()

                total_rec_loss += rec_loss.mean().item()
                total_kl_loss += kl_loss.mean().item()

        val_mse = total_rec_loss / len(dataloader)
        val_kl = total_kl_loss / len(dataloader)
        return val_mse, val_kl

    def _valid(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
        val_mse, val_kl = self.evaluate(dataloader, verbose)
        if verbose:
            print(f"epoch {epoch_id}/val_mse: {val_mse:.4f}")
        valid_metrics = {
            "callback_metric": val_mse,
            "logged_metric": {
                "val_mse": val_mse,
                "val_kl": val_kl,
            },
        }
        return valid_metrics
