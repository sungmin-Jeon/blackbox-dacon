"""Optimizer and learning-rate scheduler builders for Stage 1."""

from __future__ import annotations

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def build_optimizer(
    model: nn.Module,
    *,
    name: str = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-2,
) -> Optimizer:
    """Build an optimizer from a small, explicit set of choices."""
    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than zero")
    if weight_decay < 0:
        raise ValueError("weight_decay cannot be negative")

    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("Model has no trainable parameters")

    normalized_name = name.lower()
    if normalized_name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    if normalized_name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(
    optimizer: Optimizer,
    *,
    name: str = "cosine",
    epochs: int,
) -> LRScheduler | None:
    """Build an epoch-based scheduler, or return None when disabled."""
    normalized_name = name.lower()
    if normalized_name == "none":
        return None
    if normalized_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs),
        )

    raise ValueError(f"Unsupported scheduler: {name}")
