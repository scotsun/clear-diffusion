import torch
import torch.nn as nn
import torch.optim as optim
import mlflow

from torch.utils.data import DataLoader
from tqdm import tqdm
from mlflow.models import ModelSignature
from sklearn.metrics import roc_auc_score, average_precision_score

from src.modules.distribution import IsotropicNormalDistribution
from src.modules.metrics import accuracy_score
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
            "cls_opt": optim.AdamW(model.parameters(), lr=args["cls_lr"]),
        }
        self.ce = nn.CrossEntropyLoss()

    @torch.no_grad()
    def _get_content_feature(self, x: torch.Tensor):
        posterior: IsotropicNormalDistribution = self.vae(x)[1]
        z_feature, _ = posterior.mu.split_with_sizes(self.args["channel_split"], dim=1)
        z_feature = z_feature.reshape(x.shape[0], -1)
        return z_feature

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
                cur_step = epoch_id * len(dataloader) + batch_id

                x, y = batch["image"].to(device), batch["label"].to(device)
                if self.transform:
                    x = self.transform(x)

                z_feature = self._get_content_feature(x)

                opt.zero_grad()
                logits = self.model(z_feature)
                loss = self.ce(logits, y)
                loss.backward()
                opt.step()

                metrics = {"ce_loss": loss.item()}

                # update progress bar
                bar.set_postfix(metrics)
                if cur_step % 10 == 0:
                    mlflow.log_metrics(metrics, step=cur_step)
        return

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
        self.model.eval()
        device = self.device

        all_y = []
        all_probs = []
        all_groups = []
        with tqdm(dataloader, unit="batch", mininterval=0, disable=not verbose) as bar:
            for batch in bar:
                x, y, groups = (
                    batch["image"].to(device),
                    batch["label"].to(device),
                    batch["style"],
                )
                z_feature = self._get_content_feature(x)
                logits = self.model(z_feature)
                probs = torch.softmax(logits, dim=1)
                all_y.append(y.cpu())
                all_probs.append(probs.cpu())
                all_groups.append(groups.cpu())

        all_y = torch.cat(all_y)
        all_probs = torch.cat(all_probs)
        all_groups = torch.cat(all_groups)

        acc = accuracy_score(all_y, torch.argmax(all_probs, axis=1))
        auroc = roc_auc_score(all_y, all_probs[:, 1])
        ap = average_precision_score(all_y, all_probs[:, 1])
        return acc, auroc, ap
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

    def _valid(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
        acc, auroc, ap = self.evaluate(dataloader, verbose, epoch_id)
        if verbose:
            print(f"val_acc: {acc:.4f}, val_auroc: {auroc:.4f}, val_ap: {ap:.4f}")
        valid_metrics = {
            "callback_metric": auroc,
            "logged_metric": {
                "val_acc": acc,
                "val_auroc": auroc,
                "val_ap": ap,
            },
        }
        return valid_metrics


class DownstreamLAMTrainer:
    pass
