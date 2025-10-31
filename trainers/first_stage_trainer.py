import torch
import torch.nn as nn
import torch.nn.functional as F
import mlflow

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from mlflow.models import ModelSignature

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
        model_signature: ModelSignature,
        hyperparams: dict,
        transform=None,
    ) -> None:
        super().__init__(
            model,
            optimizer,
            early_stopping,
            verbose_period,
            device,
            model_signature,
            transform,
        )
        self.hyperparams = hyperparams

    def _train(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
        vae: VAE = self.model
        vae.train()
        opt = self.optimizer
        device = self.device

        with tqdm(dataloader, unit="batch", mininterval=0, disable=not verbose) as bar:
            bar.set_description(f"Epoch {epoch_id}")
            for batch_id, batch in enumerate(bar):
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

                bar.set_postfix(
                    rec_loss=rec_loss.mean().item(),
                    kl_loss=kl_loss.mean().item(),
                )

                cur_step = epoch_id * len(dataloader) + batch_id
                if cur_step % 10 == 0:
                    mlflow.log_metrics(
                        {
                            "rec_loss": rec_loss.mean().item(),
                            "kl_loss": kl_loss.mean().item(),
                        },
                        step=cur_step,
                    )
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


class CLEAR_VAEFirstStageTrainer(Trainer):
    def __init__(
        self,
        contrastive_criterion: nn.Module,
        model: nn.Module,
        optimizer: Optimizer,
        early_stopping: EarlyStopping | None,
        verbose_period: int,
        device: torch.device,
        model_signature: ModelSignature,
        hyperparams: dict,
        transform=None,
    ) -> None:
        super().__init__(
            model,
            optimizer,
            early_stopping,
            verbose_period,
            device,
            model_signature,
            transform,
        )
        self.contrastive_criterion = contrastive_criterion
        self.hyperparams = hyperparams

    def _train(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
        vae: VAE = self.model
        vae.train()
        opt = self.optimizer
        device = self.device

        beta = self.hyperparams["beta"]
        gamma = self.hyperparams["gamma"]

        with tqdm(dataloader, unit="batch", mininterval=0, disable=not verbose) as bar:
            bar.set_description(f"Epoch {epoch_id}")
            for batch_id, batch in enumerate(bar):
                x, y = batch["image"].to(device), batch["label"].to(device)
                if self.transform:
                    x = self.transform(x)
                opt.zero_grad()
                xhat, posterior = vae(x)
                z_c, z_s = posterior.sample().chunk(2, dim=1)

                rec_loss = F.mse_loss(xhat, x, reduction="none").sum(dim=(1, 2, 3))
                kl_loss = posterior.kl()
                con_loss = self.contrastive_criterion(z_c, y)
                ps_loss = self.contrastive_criterion(z_s, y, ps=True)
                ps_loss = ps_loss.mean()

                loss = (
                    rec_loss.mean()
                    + beta * kl_loss.mean()
                    + gamma * con_loss.mean()
                    + gamma * ps_loss.mean()
                )
                loss.backward()
                opt.step()

                bar.set_postfix(
                    rec_loss=rec_loss.mean().item(),
                    kl_loss=kl_loss.mean().item(),
                    con_loss=con_loss.mean().item(),
                    ps_loss=ps_loss.mean().item(),
                )

                cur_step = epoch_id * len(dataloader) + batch_id
                if cur_step % 10 == 0:
                    mlflow.log_metrics(
                        {
                            "rec_loss": rec_loss.mean().item(),
                            "kl_loss": kl_loss.mean().item(),
                            "con_loss": con_loss.mean().item(),
                            "ps_loss": ps_loss.mean().item(),
                        },
                        step=cur_step,
                    )
        return

    def evaluate(self, dataloader: DataLoader, verbose: bool):
        vae: VAE = self.model
        vae.eval()
        device = self.device

        losses = torch.zeros(4)
        with torch.no_grad():
            for batch in tqdm(
                dataloader, unit="batch", mininterval=0, disable=not verbose
            ):
                x, y = batch["image"].to(device), batch["label"].to(device)
                if self.transform:
                    x = self.transform(x)
                xhat, posterior = vae(x)
                z_c, z_s = posterior.sample().chunk(2, dim=1)

                rec_loss = F.mse_loss(xhat, x, reduction="none").sum(dim=(1, 2, 3))
                kl_loss = posterior.kl()
                con_loss = self.contrastive_criterion(z_c, y)
                ps_loss = self.contrastive_criterion(z_s, y, ps=True)

                losses[0] += rec_loss.mean().item()
                losses[1] += kl_loss.mean().item()
                losses[2] += con_loss.mean().item()
                losses[3] += ps_loss.mean().item()

        val_rec = losses[0] / len(dataloader)
        val_kl = losses[1] / len(dataloader)
        val_con = losses[2] / len(dataloader)
        val_ps = losses[3] / len(dataloader)
        return val_rec, val_kl, val_con, val_ps

    def _valid(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
        val_rec, val_kl, val_con, val_ps = self.evaluate(dataloader, verbose)
        if verbose:
            print(f"epoch {epoch_id}/val_rec: {val_rec:.4f}")
        valid_metrics = {
            "callback_metric": val_rec,
            "logged_metric": {
                "val_rec": val_rec,
                "val_kl": val_kl,
                "val_con": val_con,
                "val_ps": val_ps,
            },
        }
        return valid_metrics
