import torch
import torch.nn as nn
import mlflow

from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from torch import GradScaler
from mlflow.models import ModelSignature


class EarlyStopping:
    def __init__(self, patience: int, min_delta: int = 0, mode: str = "min") -> None:
        self.patience = patience
        self.counter = 0
        self.best_score = float("inf") if mode == "min" else -float("inf")
        self.early_stop = False

        match mode:
            case "min":
                self.monitor_op, self.delta_op = lambda a, b: a < b, -1 * min_delta
            case "max":
                self.monitor_op, self.delta_op = lambda a, b: a > b, min_delta
            case _:
                raise ValueError("mode must be either `min` or `max`")

    def step(self, metric_val, verbose) -> bool:
        """Return if the model state should be saved."""
        if self.monitor_op(metric_val, self.best_score + self.delta_op):
            self.best_score = metric_val
            self.counter = 0
            if verbose:
                print("[INFO]: track model state")
            return True
        else:
            self.counter += 1
            if self.counter > self.patience:
                self.early_stop = True
        return False


class Trainer(ABC):
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
        self.model = model.to(device)
        self.early_stopping = early_stopping
        self.verbose_period = verbose_period
        self.device = device
        self.model_signature = model_signature
        self.scaler = GradScaler(device=device)
        self.transform = transform
        self.model_state = model.state_dict().copy()
        self.args = args

    def _stage_model_state(self):
        self.best_model_state = self.model.state_dict().copy()

    def _log_best_model(self):
        """helper function to log model."""
        model = self.model
        model.load_state_dict(self.best_model_state)
        mlflow.pytorch.log_model(
            model,
            name="best_model",
            pip_requirements=["torch>=2.7.1+cu128"],
            signature=self.model_signature,
        )
        print("[INFO]: log best model")

    @abstractmethod
    def _configure_opts(self, args: dict):
        return dict()

    def fit(
        self,
        epochs: int,
        train_loader: DataLoader,
        valid_loader: None | DataLoader = None,
    ):
        if self.early_stopping and not valid_loader:
            raise ValueError("early_stopping requires valid_loader")
        for epoch in range(epochs):
            verbose = (epoch % self.verbose_period) == 0
            self._train(train_loader, verbose, epoch)
            if valid_loader:
                valid_metric = self._valid(valid_loader, verbose, epoch)
                mlflow.log_metrics(valid_metric["logged_metric"], step=epoch)
                if self.early_stopping:
                    track_state = self.early_stopping.step(
                        valid_metric["callback_metric"], verbose
                    )
                    if track_state:
                        self._stage_model_state()

            if self.early_stopping and self.early_stopping.early_stop:
                self._log_best_model()
                return
        self._log_best_model()
        return

    @abstractmethod
    def evaluate(self, **kwarg):
        pass

    @abstractmethod
    def _train(self, **kwarg):
        pass

    @abstractmethod
    def _valid(self, **kwarg):
        pass
