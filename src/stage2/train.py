"""Train the Stage 2 ResNet18 + BiGRU baseline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from src.common.runtime import default_device, set_seed
from src.common.video import video_frames
from src.stage2.model import Stage2Temporal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Stage 2 folder containing labels.csv")
    parser.add_argument("--model-dir", type=Path, required=True, help="Stage 2 checkpoint output folder")
    parser.add_argument("--epochs", type=int, default=int(os.getenv("EPOCHS", "1")))
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args()


def resnet_backbone() -> nn.Module:
    try:
        return resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    except Exception:
        print("Warning: ImageNet weights could not be downloaded; using weights=None.")
        return resnet18(weights=None)


def train(data_dir: Path, model_dir: Path, device: torch.device, epochs: int) -> tuple[Path, Path]:
    labels_path = data_dir / "labels.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing Stage 2 labels: {labels_path}")
    model_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(labels_path)

    backbone = resnet_backbone()
    backbone_path = model_dir / "resnet18-f37072fd.pth"
    torch.save(backbone.state_dict(), backbone_path)
    backbone.fc = nn.Identity()
    backbone.to(device).eval()
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()

    sequences = []
    with torch.inference_mode():
        for row in frame.itertuples():
            frames = video_frames(data_dir / row.path)
            batches = []
            for start in range(0, len(frames), 64):
                inputs = torch.stack([transform(Image.fromarray(image)) for image in frames[start : start + 64]])
                batches.append(backbone(inputs.to(device)).float().cpu())
            target = min(int(row.t_collision), len(frames) - 1)
            sequences.append((torch.cat(batches), target))

    temporal = Stage2Temporal().to(device)
    optimizer = torch.optim.AdamW(temporal.parameters(), 2e-4)
    for _ in range(max(1, epochs)):
        temporal.train()
        for sequence, target in sequences:
            collision, _, _ = temporal.logits(sequence[None].to(device))
            target_tensor = torch.tensor([target], device=device)
            loss = nn.functional.cross_entropy(collision, target_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    checkpoint_path = model_dir / "best.pt"
    torch.save({"model": temporal.state_dict()}, checkpoint_path)
    return checkpoint_path, backbone_path


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    set_seed(args.seed)
    device = default_device()
    print(f"device: {device}")
    checkpoint, backbone = train(data_dir, model_dir, device, args.epochs)
    print(f"Stage 2 complete: {checkpoint}")
    print(f"Stage 2 backbone: {backbone}")


if __name__ == "__main__":
    main()
