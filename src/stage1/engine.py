"""Training and validation loops for Stage 1."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.stage1.metrics import ClassificationMetrics, classification_metrics


@dataclass(frozen=True)
class ValidationResult:
    loss: float
    metrics: ClassificationMetrics


def _set_training_mode(model: nn.Module) -> None:
    """Train unfrozen modules while keeping fully frozen submodules in eval mode."""
    model.train()
    for module in model.modules():
        parameters = tuple(module.parameters(recurse=True))
        if parameters and not any(
            parameter.requires_grad for parameter in parameters
        ):
            module.eval()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    *,
    amp: bool = True,
) -> float:
    """Train for one epoch and return sample-weighted mean loss."""
    _set_training_mode(model)
    total_loss = 0.0
    total_samples = 0
    amp_enabled = amp and device.type == "cuda"

    progress = tqdm(loader, desc="train", leave=False)
    for clips, labels in progress:
        if clips.ndim != 5:
            raise ValueError(
                f"Training expects one temporal view [B,C,T,H,W], got {tuple(clips.shape)}"
            )
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(clips)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        total_loss += loss.detach().item() * batch_size
        total_samples += batch_size
        progress.set_postfix(loss=f"{loss.detach().item():.4f}")

    if total_samples == 0:
        raise ValueError("Training DataLoader produced no samples")
    return total_loss / total_samples


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    amp: bool = True,
) -> ValidationResult:
    """Evaluate the full validation split without changing model weights."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    targets: list[int] = []
    predictions: list[int] = []
    amp_enabled = amp and device.type == "cuda"

    progress = tqdm(loader, desc="val", leave=False)
    for clips, labels in progress:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            if clips.ndim == 5:
                logits = model(clips)
                loss = criterion(logits, labels)
                predicted = logits.argmax(dim=1)
            elif clips.ndim == 6:
                views = clips.size(1)
                # Run views sequentially so two-view validation does not double
                # the peak MViT batch memory.
                view_log_probabilities = torch.stack(
                    [
                        F.log_softmax(model(clips[:, view]).float(), dim=1)
                        for view in range(views)
                    ],
                    dim=1,
                )
                # Match submission inference: arithmetic mean of per-view probabilities.
                log_probabilities = torch.logsumexp(
                    view_log_probabilities, dim=1,
                ) - math.log(views)
                loss = F.nll_loss(log_probabilities, labels)
                predicted = log_probabilities.argmax(dim=1)
            else:
                raise ValueError(
                    "Validation expects [B,C,T,H,W] or [B,V,C,T,H,W], "
                    f"got {tuple(clips.shape)}"
                )

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        targets.extend(labels.cpu().tolist())
        predictions.extend(predicted.cpu().tolist())

    if total_samples == 0:
        raise ValueError("Validation DataLoader produced no samples")

    return ValidationResult(
        loss=total_loss / total_samples,
        metrics=classification_metrics(targets, predictions),
    )
