"""Evaluate a Stage 1 checkpoint on labeled source and recaptured MP4 files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.video import mvit_v2_s
from tqdm.auto import tqdm

from inference import _clip_ids, _decode_stage1_clip
from src.common.runtime import default_device
from src.stage1.metrics import classification_metrics


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".3gp", ".3gpp", ".wmv"}
LABEL_TO_NAME = {0: "ORIGINAL", 1: "RERECORDED"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root containing source/, recaptured/, and optionally source_mapping.csv",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Stage 1 best.pt produced by train_stage1.py",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA mixed precision when a GPU is available",
    )
    return parser.parse_args()


def _video_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )


def _source_id(path: Path, true_label: int) -> str:
    if true_label == 0:
        return path.stem
    return path.stem.split("_", 1)[0]


def build_samples(data_dir: Path) -> list[dict]:
    source_dir = data_dir / "source"
    recaptured_dir = data_dir / "recaptured"
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Missing source directory: {source_dir}")
    if not recaptured_dir.is_dir():
        raise FileNotFoundError(f"Missing recaptured directory: {recaptured_dir}")

    samples = []
    for path in _video_paths(source_dir):
        samples.append(
            {
                "path": path,
                "source_id": _source_id(path, 0),
                "true_label": 0,
                "capture_environment": "SOURCE",
            }
        )
    for path in _video_paths(recaptured_dir):
        samples.append(
            {
                "path": path,
                "source_id": _source_id(path, 1),
                "true_label": 1,
                "capture_environment": path.parent.name,
            }
        )

    if not samples:
        raise ValueError(f"No evaluation videos found under: {data_dir}")
    return samples


class Stage1VideoDataset(Dataset):
    def __init__(self, samples: list[dict], *, frames: int, size: int) -> None:
        self.samples = samples
        self.frames = frames
        self.size = size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path = self.samples[index]["path"]
        frame_ids = _clip_ids(path, self.frames, slot=0, slots=1)
        clip = _decode_stage1_clip(path, self.size, frame_ids)
        return clip, index


def _mapping_metadata(data_dir: Path) -> dict[str, dict]:
    mapping_path = data_dir / "source_mapping.csv"
    if not mapping_path.is_file():
        return {}

    mapping = pd.read_csv(mapping_path, dtype={"source_id": str})
    if "source_id" not in mapping.columns:
        raise ValueError(f"source_mapping.csv has no source_id column: {mapping_path}")
    if mapping["source_id"].duplicated().any():
        duplicates = sorted(mapping.loc[mapping["source_id"].duplicated(), "source_id"].unique())
        raise ValueError(f"Duplicate source_id values in source_mapping.csv: {duplicates}")

    return mapping.set_index("source_id").to_dict(orient="index")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if args.num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be between zero and one")

    data_dir = args.data_dir.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    for key in ("model", "frames", "size"):
        if key not in checkpoint:
            raise KeyError(f"Checkpoint has no {key!r} key: {checkpoint_path}")

    frames = int(checkpoint["frames"])
    size = int(checkpoint["size"])
    device = default_device()
    amp_enabled = args.amp and device.type == "cuda"

    # Match the offline submission path: create the architecture without downloading weights.
    model = mvit_v2_s(weights=None)
    model.head[1] = nn.Linear(model.head[1].in_features, 2)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()

    samples = build_samples(data_dir)
    dataset = Stage1VideoDataset(samples, frames=frames, size=size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    probabilities: list[float | None] = [None] * len(samples)
    with torch.inference_mode():
        for clips, indices in tqdm(loader, desc="evaluate"):
            clips = clips.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(clips)
            rerecorded_probabilities = torch.softmax(logits.float(), dim=1)[:, 1]
            for index, probability in zip(indices.tolist(), rerecorded_probabilities.cpu().tolist()):
                probabilities[index] = float(probability)

    if any(probability is None for probability in probabilities):
        raise RuntimeError("Some evaluation videos did not receive a prediction")

    metadata = _mapping_metadata(data_dir)
    rows = []
    targets = []
    predictions = []
    for sample, probability_value in zip(samples, probabilities):
        probability = float(probability_value)
        predicted_label = int(probability >= args.threshold)
        true_label = int(sample["true_label"])
        targets.append(true_label)
        predictions.append(predicted_label)
        source_metadata = metadata.get(sample["source_id"], {})
        rows.append(
            {
                "video_id": sample["path"].stem,
                "source_id": sample["source_id"],
                "kind": "source" if true_label == 0 else "recaptured",
                "capture_environment": sample["capture_environment"],
                "road_type": source_metadata.get("road_type", ""),
                "original_filename": source_metadata.get("original_filename", ""),
                "true_label": LABEL_TO_NAME[true_label],
                "predicted_label": LABEL_TO_NAME[predicted_label],
                "rerecorded_probability": probability,
                "correct": true_label == predicted_label,
                "path": str(sample["path"]),
            }
        )

    result = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    metrics = classification_metrics(targets, predictions)
    confusion = pd.crosstab(
        result["true_label"],
        result["predicted_label"],
        rownames=["true"],
        colnames=["predicted"],
        dropna=False,
    )
    probability_summary = result.groupby("kind")["rerecorded_probability"].agg(
        ["count", "mean", "min", "max"]
    )

    print(f"device: {device}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"videos: {len(result)}")
    print(f"frames/size: {frames}/{size}")
    print(f"accuracy: {metrics.accuracy:.4f}")
    print(f"Macro-F1: {metrics.macro_f1:.4f}")
    print(f"ORIGINAL F1: {metrics.original_f1:.4f}")
    print(f"RERECORDED F1: {metrics.rerecorded_f1:.4f}")
    print(
        "prediction ratio O/R: "
        f"{metrics.predicted_original_ratio:.3f}/"
        f"{metrics.predicted_rerecorded_ratio:.3f}"
    )
    print("\nConfusion matrix:")
    print(confusion.to_string())
    print("\nRERECORDED probability by kind:")
    print(probability_summary.to_string())
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
