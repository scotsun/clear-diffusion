import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import mlflow

from torch.utils.data import DataLoader
from tqdm import tqdm
from mlflow.models import ModelSignature

from sd_vae.ae import VAE
from modules.loss import d_hinge_loss, g_hinge_loss
from modules.loss import SupCon, SNN, DenseSupCon  # noqa
from . import EarlyStopping, Trainer


class VAEFirstStageTrainer(Trainer):
    def __init__(
        self,
        model: VAE,
        discriminator: nn.Module | None,
        early_stopping: EarlyStopping | None,
        verbose_period: int,
        device: torch.device,
        model_signature: ModelSignature,
        args: dict,
        transform=None,
    ) -> None:
        super().__init__(
            model,
            early_stopping,
            verbose_period,
            device,
            model_signature,
            args,
            transform,
        )
        # TODO: add lpips
        self.discriminator = discriminator.to(device) if discriminator else None
        if self.discriminator:
            self.discriminator.apply(self.discriminator.weight_init)
            self.logvar = nn.Parameter(torch.zeros(size=()))
        self.opts = self._configure_opts(args)

    def _configure_opts(self, args: dict):
        if self.discriminator is None:
            vae_opt = optim.Adam(self.model.parameters(), lr=args["vae_lr"])
            return {"vae_opt": vae_opt}
        else:
            vae_opt = optim.Adam(
                list(self.model.parameters()) + [self.logvar], lr=args["vae_lr"]
            )
            disc_opt = optim.Adam(self.discriminator.parameters(), lr=args["disc_lr"])
            return (
                {"vae_opt": vae_opt, "disc_opt": disc_opt}
                if self.discriminator
                else {"vae_opt": vae_opt}
            )

    def disc_factor(self, cur_step: int):
        return 1 if cur_step >= self.args["disc_warmup"] else 0

    def adaptive_d_weight(self, rec_loss, g_loss):
        last_layer = self.model.decoder.conv_out.weight
        nll_loss = (
            rec_loss / self.logvar.exp() + self.logvar
        )  # update to factor of 2 and additive constant
        nll_grad = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grad = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        d_weight = torch.norm(nll_grad) / (torch.norm(g_grad) + 1e-8)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight

    def _train(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
        has_disc = self.discriminator is not None

        vae: VAE = self.model
        vae.train()
        vae_opt = self.opts["vae_opt"]
        if has_disc:
            disc_opt = self.opts["disc_opt"]
        device = self.device

        beta = self.args["beta"]

        with tqdm(dataloader, unit="batch", mininterval=0, disable=not verbose) as bar:
            bar.set_description(f"Epoch {epoch_id}")

            for batch_id, batch in enumerate(bar):
                cur_step = epoch_id * len(dataloader) + batch_id
                x = batch["image"].to(device)
                if self.transform:
                    x = self.transform(x)
                xhat, posterior = vae(x)

                rec_loss = (
                    F.mse_loss(xhat, x, reduction="none").sum(dim=(1, 2, 3)).mean()
                )
                kl_loss = posterior.kl().mean()
                loss = rec_loss + beta * kl_loss
                metrics = {
                    "rec_loss": rec_loss.item(),
                    "kl_loss": kl_loss.item(),
                }
                if has_disc:
                    disc_factor = self.disc_factor(cur_step)
                    g_loss = g_hinge_loss(fake_logits=self.discriminator(xhat))

                    d_weight = self.adaptive_d_weight(rec_loss, g_loss)
                    loss = loss + disc_factor * d_weight * g_loss
                    metrics.update(
                        {"g_loss": g_loss.item(), "d_weight": d_weight.item()}
                    )

                vae_opt.zero_grad()
                loss.backward()
                vae_opt.step()

                if has_disc:
                    d_loss = disc_factor * d_hinge_loss(
                        real_logits=self.discriminator(x),
                        fake_logits=self.discriminator(xhat.detach()),
                    )  # no gradient path from D → G.
                    metrics.update({"d_loss": d_loss.item()})
                    disc_opt.zero_grad()
                    d_loss.backward()
                    disc_opt.step()

                # update progress bar
                bar.set_postfix(metrics)
                if cur_step % 10 == 0:
                    mlflow.log_metrics(metrics, step=cur_step)
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
                rec_loss = (
                    F.mse_loss(xhat, x, reduction="none").sum(dim=(1, 2, 3)).mean()
                )
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
        model: nn.Module,
        early_stopping: EarlyStopping | None,
        verbose_period: int,
        device: torch.device,
        model_signature: ModelSignature,
        args: dict,
        transform=None,
    ) -> None:
        super().__init__(
            model,
            early_stopping,
            verbose_period,
            device,
            model_signature,
            args,
            transform,
        )
        self._configure_contrastive_criterion(args)
        self._configure_opts(args)
        self.args = args

    def _configure_contrastive_criterion(self, args: dict):
        args = args["contrastive_module"]
        contrastive_criterions = {
            "global": {
                "content": eval(f"{args['contrastive_method']}")(
                    temperature=args["temperature"][0],
                    learnable_temp=args["learnable_temp"],
                    pool=args["pool"],
                    use_proj=args["use_proj"],
                ).to(self.device),
                "style": eval(f"{args['contrastive_method']}")(
                    temperature=args["temperature"][1],
                    learnable_temp=args["learnable_temp"],
                    pool=args["pool"],
                    use_proj=args["use_proj"],
                ).to(self.device),
            }
        }
        if args.get("use_dense"):
            contrastive_criterions["dense"] = {
                # "content": eval(f"Dense{args['contrastive_method']}")(
                "content": DenseSupCon(
                    args["temperature"][0],
                    args["learnable_temp"],
                ).to(self.device),
                "style": DenseSupCon(
                    args["temperature"][1],
                    args["learnable_temp"],
                ).to(self.device),
            }
        self.contrastive_criterions = contrastive_criterions

    def _configure_opts(self, args: dict):
        param_list = (
            list(self.model.parameters())
            + list(self.contrastive_criterions["global"]["content"].parameters())
            + list(self.contrastive_criterions["global"]["style"].parameters())
        )
        if "dense" in self.contrastive_criterions:
            param_list += list(
                self.contrastive_criterions["dense"]["content"].parameters()
            ) + list(self.contrastive_criterions["dense"]["style"].parameters())

        vae_opt = optim.Adam(param_list, lr=args["vae_lr"])
        self.opts = {"vae_opt": vae_opt}

    def _train(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
        vae: VAE = self.model
        vae.train()
        opt = self.opts["vae_opt"]
        device = self.device

        channel_split = self.args["channel_split"]
        beta = self.args["beta"]
        gamma = self.args["gamma"]

        with tqdm(dataloader, unit="batch", mininterval=0, disable=not verbose) as bar:
            bar.set_description(f"Epoch {epoch_id}")
            for batch_id, batch in enumerate(bar):
                x, y = batch["image"].to(device), batch["label"].to(device)
                if self.transform:
                    x = self.transform(x)
                opt.zero_grad()
                xhat, posterior = vae(x)
                z_c, z_s = posterior.sample().split_with_sizes(channel_split, dim=1)

                rec_loss = (
                    F.mse_loss(xhat, x, reduction="none").sum(dim=(1, 2, 3)).mean()
                )
                kl_loss = posterior.kl().mean()
                con_loss = self.contrastive_criterions["global"]["content"](
                    z_c, y
                ).mean()
                ps_loss = self.contrastive_criterions["global"]["style"](
                    z_s, y, ps=True
                ).mean()

                if "dense" in self.contrastive_criterions:
                    dense_con_loss = self.contrastive_criterions["dense"]["content"](
                        z_c, y
                    ).mean()
                    dense_ps_loss = self.contrastive_criterions["dense"]["style"](
                        z_s, y, ps=True
                    ).mean()
                    loss = loss = (
                        rec_loss
                        + beta * kl_loss
                        + gamma * (0.5 * con_loss + 0.5 * dense_con_loss)
                        + gamma * (0.5 * ps_loss + 0.5 * dense_ps_loss)
                    )
                else:
                    dense_con_loss = torch.tensor(0.0, device=device)
                    dense_ps_loss = torch.tensor(0.0, device=device)
                    loss = rec_loss + beta * kl_loss + gamma * con_loss + 0 * ps_loss

                loss.backward()
                opt.step()

                bar.set_postfix(
                    rec_loss=rec_loss.item(),
                    kl_loss=kl_loss.item(),
                    con_loss=con_loss.item(),
                    ps_loss=ps_loss.item(),
                    dense_con_loss=dense_con_loss.item(),
                    dense_ps_loss=dense_ps_loss.item(),
                )

                cur_step = epoch_id * len(dataloader) + batch_id
                if cur_step % 50 == 0:
                    mlflow.log_metrics(
                        {
                            "rec_loss": rec_loss.item(),
                            "kl_loss": kl_loss.item(),
                            "con_loss": con_loss.item(),
                            "ps_loss": ps_loss.item(),
                            "dense_con_loss": dense_con_loss.item(),
                            "dense_ps_loss": dense_ps_loss.item(),
                        },
                        step=cur_step,
                    )
        return

    def evaluate(self, dataloader: DataLoader, verbose: bool):
        vae: VAE = self.model
        vae.eval()
        device = self.device
        channel_split = self.args["channel_split"]

        losses = {
            "rec_loss": 0.0,
            "kl_loss": 0.0,
            "con_loss": 0.0,
            "ps_loss": 0.0,
            "dense_con_loss": 0.0,
            "dense_ps_loss": 0.0,
        }
        with torch.no_grad():
            for batch in tqdm(
                dataloader, unit="batch", mininterval=0, disable=not verbose
            ):
                x, y = batch["image"].to(device), batch["label"].to(device)
                if self.transform:
                    x = self.transform(x)
                xhat, posterior = vae(x)
                z_c, z_s = posterior.sample().split_with_sizes(channel_split, dim=1)

                rec_loss = (
                    F.mse_loss(xhat, x, reduction="none").sum(dim=(1, 2, 3)).mean()
                )
                kl_loss = posterior.kl().mean()
                con_loss = self.contrastive_criterions["global"]["content"](
                    z_c, y
                ).mean()
                ps_loss = self.contrastive_criterions["global"]["style"](
                    z_s, y, ps=True
                ).mean()

                losses["rec_loss"] += rec_loss.item()
                losses["kl_loss"] += kl_loss.item()
                losses["con_loss"] += con_loss.item()
                losses["ps_loss"] += ps_loss.item()

                if "dense" in self.contrastive_criterions:
                    dense_con_loss = self.contrastive_criterions["dense"]["content"](
                        z_c, y
                    ).mean()
                    dense_ps_loss = self.contrastive_criterions["dense"]["style"](
                        z_s, y, ps=True
                    ).mean()
                    losses["dense_con_loss"] += dense_con_loss.item()
                    losses["dense_ps_loss"] += dense_ps_loss.item()

        for k in losses:
            losses[k] /= len(dataloader)

        return losses

    def _valid(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
        val_losses = self.evaluate(dataloader, verbose)
        val_rec = val_losses["rec_loss"]
        if verbose:
            print(f"epoch {epoch_id}/val_rec: {val_rec:.4f}")

        logged_metric = {f"val_{k}": v for k, v in val_losses.items()}
        valid_metrics = {
            "callback_metric": val_rec,
            "logged_metric": logged_metric,
        }
        return valid_metrics
