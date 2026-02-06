import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import mlflow

from torch.utils.data import DataLoader
from tqdm import tqdm
from mlflow.models import ModelSignature

from src.modules.distribution import IsotropicNormalDistribution
from . import EarlyStopping, Trainer


class DownstreamMLPTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        vae: nn.Module,
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
        self.vae = vae
        self.args = args
        self.opts = {
            "cls_opt": optim.Adam(model.parameters(), lr=args["cls_lr"]),
        }

    def _unpack_batch(self, batch):
        if isinstance(batch, dict):
            if "x" in batch:
                x = batch["x"]
            elif "image" in batch:
                x = batch["image"]
            elif "data" in batch:
                x = batch["data"]
            else:
                x = list(batch.values())[0]

            if "y" in batch:
                y = batch["y"]
            elif "label" in batch:
                y = batch["label"]
            elif "target" in batch:
                y = batch["target"]
            else:
                y = list(batch.values())[1]

        elif isinstance(batch, (list, tuple)):
            x, y = batch[0], batch[1]
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        return x, y

    def _get_vae_feature(self, x):
        moments = self.vae.encoder(x)

        posterior = IsotropicNormalDistribution(moments)

        if hasattr(posterior, "mean"):
            z = posterior.mean
        elif hasattr(posterior, "mode"):
            z = posterior.mode()
        else:
            c = moments.shape[1] // 2
            z = moments[:, :c, :, :]

        z = z.reshape(z.shape[0], -1)

        return z

    def _train(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
        model: nn.Module = self.model
        model.train()
        vae: nn.Module = self.vae
        vae.eval()
        opt = self.opts["cls_opt"]
        device = self.device

        with tqdm(dataloader, unit="batch", mininterval=0, disable=not verbose) as bar:
            bar.set_description(f"epoch {epoch_id}")
            for batch_id, batch in enumerate(bar):
                x, y = batch["image"].to(device), batch["label"].to(device)
                if self.transform:
                    x = self.transform(x)

                ### TODO: make this a func
                with torch.no_grad():
                    _, posterior = vae(x)
                    z_feature, _ = posterior.sample().split_with_sizes(
                        self.args["channel_split"], dim=1
                    )
                    z_feature = z_feature.reshape(x.shape[0], -1)
                #######################

                self.optimizer.zero_grad()

                logits = self.model(z_feature)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()
                bar.set_postfix(loss=float(loss))

    def _valid(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
        if verbose:
            (aupr_scores, auroc_scores), acc = self.evaluate(
                dataloader, verbose, epoch_id
            )
            print(f"val_acc: {acc:.3f}")

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
        self.model.eval()
        all_y = []
        all_probs = []
        groups = []

        with tqdm(dataloader, unit="batch", mininterval=0, disable=not verbose) as bar:
            for batch in bar:
                pass
        #         x, y = self._unpack_batch(batch)

        #         g_batch = None
        #         if isinstance(batch, dict):
        #             if "c" in batch:
        #                 g_batch = batch["c"]
        #             elif "group" in batch:
        #                 g_batch = batch["group"]
        #         elif isinstance(batch, (list, tuple)) and len(batch) > 2:
        #             g_batch = batch[2]
        #         if g_batch is not None:
        #             groups.append(g_batch.cpu().numpy())

        #         y = y.reshape(-1)
        #         x = x.to(self.device)

        #         z_feature = self._get_vae_feature(x)

        #         logits = self.model(z_feature)
        #         probs = torch.softmax(logits, dim=1)

        #         all_y.append(y.cpu())
        #         all_probs.append(probs.cpu())

        #     all_y = torch.cat(all_y).numpy()
        #     all_probs = torch.cat(all_probs).numpy()

        #     if len(groups) > 0:
        #         groups = np.concatenate(groups)
        #     else:
        #         groups = np.zeros_like(all_y)

        #     acc = accuracy_score(all_y, np.argmax(all_probs, axis=1))
        #     aupr_scores = {}
        #     auroc_scores = {}

        #     unique_groups = np.unique(groups)
        #     for g in unique_groups:
        #         mask = groups == g
        #         y_sub = all_y[mask]
        #         prob_sub = all_probs[mask]

        #         if len(np.unique(y_sub)) > 1 and prob_sub.shape[1] >= 2:
        #             try:
        #                 auroc = roc_auc_score(y_sub, prob_sub[:, 1])
        #                 aupr = average_precision_score(y_sub, prob_sub[:, 1])
        #             except ValueError:
        #                 auroc, aupr = 0.5, 0.0
        #         else:
        #             auroc, aupr = 0.5, 0.0

        #         k = f"group_{g}" if isinstance(g, (int, np.integer)) else str(g)
        #         auroc_scores[k] = float(auroc)
        #         aupr_scores[k] = float(aupr)

        # return (aupr_scores, auroc_scores), acc


class DownstreamLAMTrainer:
    pass
