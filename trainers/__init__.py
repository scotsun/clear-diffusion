from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import mlflow


class EarlyStopping:
    def __init__(
        self, patience: int, min_delta: int = 0, mode: str = "min", model_signature=None
    ) -> None:
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.model_signature = model_signature

        match mode:
            case "min":
                self.monitor_op, self.delta_op = lambda a, b: a < b, -1 * min_delta
            case "max":
                self.monitor_op, self.delta_op = lambda a, b: a > b, min_delta
            case _:
                raise ValueError("mode must be either `min` or `max`")

    def _log_best_model(self, model):
        """helper function to log model."""
        mlflow.pytorch.log_model(
            model,
            "best_model",
            pip_requirements=["torch==2.2.1+cu121"],
            signature=self.model_signature,
        )

    def step(self, metric_val, model):
        # save the first chkpt
        if self.best_score is None:
            self.best_score = metric_val
            self._log_best_model(model)
            return
        # save the subsequent chkpt
        if self.monitor_op(metric_val, self.best_score + self.delta_op):
            self.best_score = metric_val
            self.counter = 0
            self._log_best_model(model)
            return
        else:
            self.counter += 1
            if self.counter > self.patience:
                self.early_stop = True
            return


class Trainer(ABC):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        early_stopping: EarlyStopping | None,
        verbose_period: int,
        device: torch.device,
        transform=None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.early_stopping = early_stopping
        self.verbose_period = verbose_period
        self.device = device
        self.transform = transform
        self.best_score = -float("inf")

    def fit(
        self,
        epochs: int,
        train_loader: DataLoader,
        valid_loader: None | DataLoader = None,
    ):
        for epoch in range(epochs):
            verbose = (epoch % self.verbose_period) == 0
            self._train(train_loader, verbose, epoch)
            if valid_loader is not None:
                self._valid(valid_loader, verbose, epoch)
                if self.early_stopping and self.early_stopping.early_stop:
                    break

    @abstractmethod
    def evaluate(self, **kwarg):
        pass

    @abstractmethod
    def _train(self, **kwarg):
        pass

    @abstractmethod
    def _valid(self, **kwarg):
        pass
