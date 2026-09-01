"""Train the Stage 1 ORIGINAL/RERECORDED baseline classifier."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from src.common.runtime import default_device, set_seed
from src.common.video import load_clip
from src.stage1.model import Stage1MViT


MEAN = torch.tensor([0.45, 0.45, 0.45])[:, None, None, None]
STD = torch.tensor([0.225, 0.225, 0.225])[:, None, None, None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Stage 1 folder containing labels.csv")
    parser.add_argument("--model-dir", type=Path, required=True, help="Stage 1 checkpoint output folder")
    parser.add_argument("--epochs", type=int, default=int(os.getenv("EPOCHS", "1")))
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args()


def train(
    data_dir: Path,
    model_dir: Path,
    device: torch.device,
    epochs: int,
    seed: int,
) -> Path:
    labels_path = data_dir / "labels.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing Stage 1 labels: {labels_path}")
    model_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(labels_path)
    model = Stage1MViT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), 1e-4)

    for epoch in range(epochs):
        model.train()
        shuffled = frame.sample(frac=1, random_state=seed + epoch)
        for row in shuffled.itertuples():
            inputs, _ = load_clip(data_dir / row.path, frames=16)
            inputs = (inputs - MEAN) / STD
            target = torch.tensor([0 if row.label == "ORIGINAL" else 1], device=device)
            loss = nn.functional.cross_entropy(model(inputs[None].to(device)), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    checkpoint_path = model_dir / "best.pt"
    torch.save({"model": model.net.state_dict(), "size": 224, "frames": 16}, checkpoint_path)
    return checkpoint_path


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    set_seed(args.seed)
    device = default_device()
    print(f"device: {device}")
    checkpoint = train(data_dir, model_dir, device, args.epochs, args.seed)
    print(f"Stage 1 complete: {checkpoint}")


if __name__ == "__main__":
    main()
