"""Extract frozen ResNet18 global features for Stage 2 collision training."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.stage2.collision_aihub_train import (
    cache_collision_features,
    load_collision_labels,
    load_collision_sample,
)


def build_manifest(labels: pd.DataFrame, feature_dir: Path) -> pd.DataFrame:
    """Validate every cache and describe exactly what will enter training."""
    rows = []
    for row in labels.itertuples(index=False):
        sample = load_collision_sample(row, feature_dir)
        cache_path = feature_dir / f"{sample['ID']}.pt"
        rows.append(
            {
                "ID": sample["ID"],
                "split": sample["split"],
                "collision_frame": sample["target_frame"],
                "target_index": sample["target_index"],
                "frame_count": len(sample["frames"]),
                "first_frame": int(sample["frames"][0]),
                "last_frame": int(sample["frames"][-1]),
                "fps": sample["fps"],
                "feature_shape": f"{len(sample['frames'])}x{sample['x'].shape[1]}",
                "cache_size_mb": cache_path.stat().st_size / 1024**2,
                "cache_path": str(cache_path),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.expected_count is not None and args.expected_count < 1:
        raise ValueError("expected-count must be positive")

    labels_path = args.labels.expanduser().resolve()
    video_root = args.video_root.expanduser().resolve()
    backbone_path = args.backbone.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    for path, name in (
        (labels_path, "labels"),
        (video_root, "video root"),
        (backbone_path, "backbone"),
    ):
        exists = path.is_dir() if name == "video root" else path.is_file()
        if not exists:
            raise FileNotFoundError(f"Missing {name}: {path}")

    labels = load_collision_labels(labels_path)
    if args.expected_count is not None and len(labels) != args.expected_count:
        raise ValueError(
            f"Expected {args.expected_count} usable collision labels, got {len(labels)}"
        )

    split_counts = labels["split"].value_counts().to_dict()
    print(f"Device: {args.device}")
    print(f"Usable collision labels: {len(labels)}")
    print(f"Split: {split_counts}")
    print("Feature: frozen ResNet18 ImageNet center crop -> [T, 512] FP32")
    print(f"Output: {output_dir}")

    cache_collision_features(
        labels=labels,
        video_root=video_root,
        feature_dir=output_dir,
        backbone_path=backbone_path,
        device=torch.device(args.device),
        batch_size=args.batch_size,
        force=args.force,
    )

    manifest = build_manifest(labels, output_dir)
    manifest_path = output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    cached_ids = set(manifest["ID"])
    expected_ids = set(labels["ID"])
    if cached_ids != expected_ids or len(manifest) != len(labels):
        raise RuntimeError("Feature manifest IDs do not exactly match usable labels")
    if (manifest["target_index"] < 0).any() or (
        manifest["target_index"] >= manifest["frame_count"]
    ).any():
        raise RuntimeError("At least one collision target is outside its feature sequence")

    print("Extraction complete")
    print(f"Videos: {len(manifest)}")
    print(f"Total cached size: {manifest['cache_size_mb'].sum():.2f} MB")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
