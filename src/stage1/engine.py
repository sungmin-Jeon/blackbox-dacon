"""Training and validation loops for Stage 1."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.stage1.metrics import ClassificationMetrics, classification_metrics


@dataclass(frozen=True)
class ValidationResult:
    loss: float
    metrics: ClassificationMetrics


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
    model.train()
    total_loss = 0.0
    total_samples = 0
    amp_enabled = amp and device.type == "cuda"

    progress = tqdm(loader, desc="train", leave=False)
    for clips, labels in progress:
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
            logits = model(clips)
            loss = criterion(logits, labels)

        predicted = logits.argmax(dim=1)
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
