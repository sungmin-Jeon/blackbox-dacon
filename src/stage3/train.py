"""Train the Stage 3 MViTv2-S multi-head baseline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from src.common.runtime import default_device, set_seed
from src.common.video import load_clip
from src.stage3.model import Stage3MViT


MEAN = torch.tensor([0.45, 0.45, 0.45])[:, None, None]
STD = torch.tensor([0.225, 0.225, 0.225])[:, None, None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Stage 3 folder containing labels.csv")
    parser.add_argument("--model-dir", type=Path, required=True, help="Stage 3 checkpoint output folder")
    parser.add_argument("--epochs", type=int, default=int(os.getenv("EPOCHS", "1")))
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args()


def train(data_dir: Path, model_dir: Path, device: torch.device, epochs: int) -> Path:
    labels_path = data_dir / "labels.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing Stage 3 labels: {labels_path}")
    model_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(labels_path)
    accel_map = {"ACCELERATING": 0, "DECELERATING": 1, "CONSTANT": 2, "STOPPED": 3}
    steer_map = {"LEFT": 0, "STRAIGHT": 1, "RIGHT": 2}
    model = Stage3MViT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), 1e-4)

    for _ in range(epochs):
        model.train()
        for row in frame.itertuples():
            inputs, _ = load_clip(data_dir / "videos" / f"{row.ID}.mp4", 16, int(row.frame_index))
            inputs = (inputs - MEAN[:, None, :, :]) / STD[:, None, :, :]
            accel, steer = model(inputs[None].to(device))
            loss = nn.functional.cross_entropy(accel, torch.tensor([accel_map[row.accel_label]], device=device))
            loss += nn.functional.cross_entropy(steer, torch.tensor([steer_map[row.steer_label]], device=device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    checkpoint_path = model_dir / "best.pt"
    torch.save({"model": model.state_dict()}, checkpoint_path)
    return checkpoint_path


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    set_seed(args.seed)
    device = default_device()
    print(f"device: {device}")
    checkpoint = train(data_dir, model_dir, device, args.epochs)
    print(f"Stage 3 complete: {checkpoint}")


if __name__ == "__main__":
    main()
