"""Predict collision context frames for directly labelled Stage 2 videos."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from tqdm.auto import tqdm


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class CollisionBiGRU(nn.Module):
    def __init__(
        self,
        input_size: int = 512,
        hidden_size: int = 192,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.collision_head = nn.Linear(hidden_size * 2, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.gru(inputs)
        return self.collision_head(self.dropout(hidden)).squeeze(-1)


class BaselineCollisionBiGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.r = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.15 if num_layers > 1 else 0.0,
        )
        self.tc = nn.Linear(hidden_size * 2, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.r(inputs)
        return self.tc(hidden).squeeze(-1)


def _checkpoint_state(checkpoint: dict) -> dict[str, torch.Tensor]:
    for key in ("model_state_dict", "model", "state_dict"):
        if key in checkpoint:
            state = checkpoint[key]
            break
    else:
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            state = checkpoint
        else:
            raise ValueError("Collision checkpoint has no model state dictionary")
    return {
        name.removeprefix("module."): value
        for name, value in state.items()
    }


def _gru_dimensions(state: dict[str, torch.Tensor], prefix: str) -> tuple[int, int, int]:
    input_key = f"{prefix}.weight_ih_l0"
    hidden_key = f"{prefix}.weight_hh_l0"
    if input_key not in state or hidden_key not in state:
        raise ValueError(f"Missing GRU weights for prefix {prefix}")
    input_size = int(state[input_key].shape[1])
    hidden_size = int(state[hidden_key].shape[1])
    layer_prefix = f"{prefix}.weight_ih_l"
    layers = [
        int(key[len(layer_prefix):])
        for key in state
        if key.startswith(layer_prefix) and key[len(layer_prefix):].isdigit()
    ]
    if not layers:
        raise ValueError(f"Cannot infer GRU layers for prefix {prefix}")
    return input_size, hidden_size, max(layers) + 1


def collision_model_from_checkpoint(checkpoint: dict) -> tuple[nn.Module, str]:
    state = _checkpoint_state(checkpoint)
    if "gru.weight_ih_l0" in state and "collision_head.weight" in state:
        input_size, hidden_size, num_layers = _gru_dimensions(state, "gru")
        model = CollisionBiGRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=float(checkpoint.get("dropout", 0.3)),
        )
        model.load_state_dict(
            {
                key: value
                for key, value in state.items()
                if key.startswith("gru.") or key.startswith("collision_head.")
            }
        )
        return model, "collision_v0"
    if "r.weight_ih_l0" in state and "tc.weight" in state:
        input_size, hidden_size, num_layers = _gru_dimensions(state, "r")
        model = BaselineCollisionBiGRU(input_size, hidden_size, num_layers)
        model.load_state_dict(
            {
                key: value
                for key, value in state.items()
                if key.startswith("r.") or key.startswith("tc.")
            }
        )
        return model, "baseline"
    raise ValueError(f"Unsupported collision checkpoint keys: {list(state)[:10]}")


def _video_index(root: Path) -> dict[str, Path]:
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    index: dict[str, Path] = {}
    duplicates = []
    for path in paths:
        if path.stem in index:
            duplicates.append(path.stem)
        index[path.stem] = path
    if duplicates:
        raise ValueError(f"Duplicate video IDs: {sorted(set(duplicates))[:10]}")
    return index


def _video_batches(path: Path, transform, batch_size: int):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise RuntimeError(f"Invalid FPS: {path}")
    frames = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(transform(Image.fromarray(rgb)))
        if len(frames) == batch_size:
            yield torch.stack(frames), fps
            frames.clear()
    capture.release()
    if frames:
        yield torch.stack(frames), fps


@torch.inference_mode()
def extract_video_features(
    path: Path,
    backbone: nn.Module,
    transform,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, float]:
    features = []
    fps = 0.0
    for images, fps in _video_batches(path, transform, batch_size):
        features.append(backbone(images.to(device, non_blocking=True)).float().cpu())
    if not features:
        raise RuntimeError(f"No decodable frames: {path}")
    sequence = torch.cat(features)
    return sequence, fps


@torch.inference_mode()
def predict_video(
    path: Path,
    backbone: nn.Module,
    temporal: CollisionBiGRU,
    transform,
    device: torch.device,
    batch_size: int,
) -> tuple[int, int, float, float]:
    sequence, fps = extract_video_features(
        path, backbone, transform, device, batch_size
    )
    logits = temporal(sequence.unsqueeze(0).to(device))
    probabilities = logits.softmax(dim=1)
    predicted_index = int(logits.argmax(dim=1).item())
    confidence = float(probabilities[0, predicted_index].item())
    return predicted_index + 1, len(sequence), fps, confidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--collision-checkpoint", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    labels_path = args.labels.expanduser().resolve()
    video_root = args.video_root.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    labels = pd.read_csv(labels_path, dtype={"ID": str})
    if "ID" not in labels:
        raise ValueError("Labels require an ID column")
    if "status" in labels:
        labels = labels[labels["status"].fillna("").str.upper().eq("KEEP")]
    if "collision_frame" not in labels and "collision_best" in labels:
        labels["collision_frame"] = labels["collision_best"]
    if labels["ID"].duplicated().any():
        raise ValueError("Labels contain duplicate IDs")
    labels = labels.reset_index(drop=True)

    videos = _video_index(video_root)
    missing = [video_id for video_id in labels["ID"] if video_id not in videos]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} videos: {missing[:10]}")

    device = torch.device(args.device)
    backbone = resnet18(weights=None)
    backbone.load_state_dict(
        torch.load(args.backbone.expanduser().resolve(), map_location="cpu", weights_only=True)
    )
    backbone.fc = nn.Identity()
    backbone.to(device).eval().requires_grad_(False)

    checkpoint = torch.load(
        args.collision_checkpoint.expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    temporal, checkpoint_format = collision_model_from_checkpoint(checkpoint)
    temporal.to(device).eval().requires_grad_(False)
    print(f"Collision checkpoint format: {checkpoint_format}")
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    completed: set[str] = set()
    if output_path.is_file() and not args.force:
        previous = pd.read_csv(output_path, dtype={"ID": str})
        if "ID" not in previous or "collision_pred_frame" not in previous:
            raise ValueError(f"Invalid existing prediction CSV: {output_path}")
        rows = previous.to_dict("records")
        completed = set(previous["ID"])
        print(f"Resume: {len(completed)} predictions already exist")

    target_by_id = labels.set_index("ID").get("collision_frame")
    for video_id in tqdm(labels["ID"], desc="Predicting collision context"):
        if video_id in completed:
            continue
        predicted_frame, frame_count, fps, confidence = predict_video(
            videos[video_id], backbone, temporal, transform, device, args.batch_size
        )
        target_value = target_by_id.get(video_id) if target_by_id is not None else None
        target_frame = (
            int(float(target_value))
            if target_value is not None and pd.notna(target_value) and str(target_value).strip()
            else None
        )
        error_frames = predicted_frame - target_frame if target_frame is not None else None
        rows.append(
            {
                "ID": video_id,
                "collision_pred_frame": predicted_frame,
                "collision_pred_index": predicted_frame - 1,
                "frame_count": frame_count,
                "fps": fps,
                "confidence": confidence,
                "target_collision_frame": target_frame,
                "error_frames": error_frames,
                "abs_error_sec": abs(error_frames) / fps if error_frames is not None else None,
            }
        )
        if len(rows) % 10 == 0:
            pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")

    predictions = pd.DataFrame(rows)
    predictions = predictions.drop_duplicates("ID", keep="last")
    predictions = predictions.set_index("ID").loc[labels["ID"]].reset_index()
    predictions.to_csv(output_path, index=False, encoding="utf-8-sig")
    evaluated = predictions.dropna(subset=["abs_error_sec"])
    print(f"Saved: {output_path}")
    print(f"Predictions: {len(predictions)}")
    if len(evaluated):
        print(f"Annotated collisions: {len(evaluated)}")
        print(f"Collision Accuracy@0.3s: {(evaluated['abs_error_sec'] <= 0.3 + 1e-9).mean():.1%}")
        print(f"Collision median error: {evaluated['abs_error_sec'].median():.3f}s")


if __name__ == "__main__":
    main()
