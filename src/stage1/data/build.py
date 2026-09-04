"""Dataset and DataLoader builders for Stage 1 training."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.stage1.data.baidu import BaiduMoireDataset


def _seed_worker(worker_id: int) -> None:
    """Give every DataLoader worker a deterministic random seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_baidu_datasets(
    data_root: str | Path,
    *,
    frames: int = 16,
    size: int = 224,
    expected_source_frames: int | None = 60,
) -> tuple[BaiduMoireDataset, BaiduMoireDataset]:
    """Build datasets from Baidu's official train/validation split."""
    data_root = Path(data_root).expanduser().resolve()

    train_dataset = BaiduMoireDataset(
        data_root / "train",
        frames=frames,
        size=size,
        expected_source_frames=expected_source_frames,
    )
    val_dataset = BaiduMoireDataset(
        data_root / "val",
        frames=frames,
        size=size,
        expected_source_frames=expected_source_frames,
    )

    return train_dataset, val_dataset


def build_baidu_dataloaders(
    data_root: str | Path,
    *,
    frames: int = 16,
    size: int = 224,
    batch_size: int = 2,
    val_batch_size: int | None = None,
    num_workers: int = 2,
    pin_memory: bool | None = None,
    seed: int = 42,
    expected_source_frames: int | None = 60,
) -> tuple[DataLoader, DataLoader]:
    """Build reproducible Baidu train and validation DataLoaders."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if val_batch_size is not None and val_batch_size <= 0:
        raise ValueError("val_batch_size must be greater than zero")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    train_dataset, val_dataset = build_baidu_datasets(
        data_root,
        frames=frames,
        size=size,
        expected_source_frames=expected_source_frames,
    )

    if val_batch_size is None:
        val_batch_size = batch_size
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=_seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=_seed_worker,
    )

    return train_loader, val_loader
