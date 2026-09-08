"""Dataset and DataLoader builders for Stage 1 training."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.stage1.data.baidu import BaiduMoireDataset
from src.stage1.data.direct import DirectStage1Dataset


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


def build_direct_datasets(
    split_csv: str | Path,
    *,
    video_root: str | Path | None = None,
    frames: int = 16,
    size: int = 224,
    cache_dir: str | Path | None = None,
) -> tuple[DirectStage1Dataset, DirectStage1Dataset]:
    """Build train/validation datasets from stage1_split.csv."""

    split_csv = Path(split_csv).expanduser().resolve()

    if video_root is not None:
        video_root = Path(video_root).expanduser().resolve()

    train_dataset = DirectStage1Dataset(
        split_csv,
        split="train",
        video_root=video_root,
        frames=frames,
        size=size,
        cache_dir=cache_dir,
    )

    val_dataset = DirectStage1Dataset(
        split_csv,
        split="val",
        video_root=video_root,
        frames=frames,
        size=size,
        cache_dir=cache_dir,
    )

    return train_dataset, val_dataset


def _balanced_sampler(
    labels: list[int],
    generator: torch.Generator,
) -> WeightedRandomSampler:
    """Sample ORIGINAL and RERECORDED with equal probability."""

    label_tensor = torch.tensor(labels, dtype=torch.long)

    class_counts = torch.bincount(
        label_tensor,
        minlength=2,
    ).float()

    if torch.any(class_counts == 0):
        raise ValueError(
            f"Both classes are required. Counts: "
            f"{class_counts.tolist()}"
        )

    class_weights = 1.0 / class_counts
    sample_weights = class_weights[label_tensor]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True,
        generator=generator,
    )


def build_direct_dataloaders(
    split_csv: str | Path,
    *,
    video_root: str | Path | None = None,
    frames: int = 16,
    size: int = 224,
    batch_size: int = 2,
    val_batch_size: int | None = None,
    num_workers: int = 2,
    pin_memory: bool | None = None,
    seed: int = 42,
    balanced_sampling: bool = True,
    cache_dir: str | Path | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Build Direct train and validation DataLoaders."""

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero"
        )

    if (
        val_batch_size is not None
        and val_batch_size <= 0
    ):
        raise ValueError(
            "val_batch_size must be greater than zero"
        )

    if num_workers < 0:
        raise ValueError(
            "num_workers cannot be negative"
        )

    train_dataset, val_dataset = build_direct_datasets(
        split_csv,
        video_root=video_root,
        frames=frames,
        size=size,
        cache_dir=cache_dir,
    )

    if val_batch_size is None:
        val_batch_size = batch_size

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    generator = torch.Generator()
    generator.manual_seed(seed)

    sampler = None

    if balanced_sampling:
        sampler = _balanced_sampler(
            train_dataset.labels,
            generator,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,

        # sampler와 shuffle은 동시에 사용할 수 없음
        shuffle=sampler is None,
        sampler=sampler,

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
