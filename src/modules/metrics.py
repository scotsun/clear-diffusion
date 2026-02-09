import torch


def accuracy_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return (y_true == y_pred).float().mean().item()
