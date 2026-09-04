"""Checkpoint and experiment-log helpers for Stage 1."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from src.stage1.engine import ValidationResult
from src.stage1.model import Stage1MViT


METRIC_COLUMNS = (
    "epoch",
    "learning_rate",
    "train_loss",
    "val_loss",
    "val_accuracy",
    "val_macro_f1",
    "original_f1",
    "rerecorded_f1",
    "predicted_original_ratio",
    "predicted_rerecorded_ratio",
)


def save_config(output_dir: str | Path, config: dict[str, Any]) -> Path:
    """Save the resolved experiment arguments as JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "config.json"
    path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def append_metrics(
    output_dir: str | Path,
    *,
    epoch: int,
    learning_rate: float,
    train_loss: float,
    validation: ValidationResult,
) -> Path:
    """Append one epoch to metrics.csv."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "metrics.csv"
    write_header = not path.exists()
    metrics = asdict(validation.metrics)
    row = {
        "epoch": epoch,
        "learning_rate": learning_rate,
        "train_loss": train_loss,
        "val_loss": validation.loss,
        "val_accuracy": metrics["accuracy"],
        "val_macro_f1": metrics["macro_f1"],
        "original_f1": metrics["original_f1"],
        "rerecorded_f1": metrics["rerecorded_f1"],
        "predicted_original_ratio": metrics["predicted_original_ratio"],
        "predicted_rerecorded_ratio": metrics["predicted_rerecorded_ratio"],
    }

    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRIC_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return path


def save_checkpoint(
    path: str | Path,
    *,
    model: Stage1MViT,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    size: int,
    frames: int,
    val_macro_f1: float,
    config: dict[str, Any],
) -> Path:
    """Save a checkpoint compatible with inference.predict_stage1."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    checkpoint = {
        # inference.py loads this into a bare torchvision MViTv2-S.
        "model": model.net.state_dict(),
        "epoch": epoch,
        "size": size,
        "frames": frames,
        "val_macro_f1": val_macro_f1,
        "class_to_label": {
            "ORIGINAL": 0,
            "RERECORDED": 1,
        },
        "config": config,
    }
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()

    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)
    return path
