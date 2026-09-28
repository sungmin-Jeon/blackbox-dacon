"""Predict collision frames from cached Stage 2 feature maps."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.stage2.collision_aihub_train import collision_metrics, load_collision_labels
from src.stage2.collision_feature_compare import (
    CollisionFeatureModel,
    load_feature_samples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    labels = load_collision_labels(args.labels.expanduser().resolve())
    feature_dir = args.feature_dir.expanduser().resolve()
    samples, feature_info = load_feature_samples(labels, feature_dir)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = CollisionFeatureModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])

    expected_kind = checkpoint["model_config"]["feature_kind"]
    expected_channels = checkpoint["model_config"]["input_channels"]
    if feature_info["feature_kind"] != expected_kind:
        raise ValueError(
            f"Feature kind mismatch: {feature_info['feature_kind']} != {expected_kind}"
        )
    if feature_info["input_channels"] != expected_channels:
        raise ValueError(
            f"Feature channels mismatch: {feature_info['input_channels']} != "
            f"{expected_channels}"
        )

    device = torch.device(args.device)
    amp = not args.no_amp and device.type == "cuda"
    model.to(device).eval()
    rows = []
    for sample in samples:
        inputs = sample["x"].unsqueeze(0).to(device=device, dtype=torch.float32)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp
        ):
            logits = model(inputs)
        predicted_index = int(logits.argmax(dim=1).item())
        predicted_frame = int(sample["frames"][predicted_index])
        error_frames = predicted_frame - sample["target_frame"]
        rows.append(
            {
                "ID": sample["ID"],
                "split": sample["split"],
                "collision_pred_frame": predicted_frame,
                "target_frame": sample["target_frame"],
                "fps": sample["fps"],
                "error_frames": error_frames,
                "abs_error_sec": abs(error_frames) / sample["fps"],
            }
        )

    predictions = pd.DataFrame(rows)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Predictions: {len(predictions)} | Saved: {output}")
    for split, frame in predictions.groupby("split"):
        metrics = collision_metrics(frame)
        print(
            f"{split}: @0.3 {metrics['acc_03']:.1%} | "
            f"median {metrics['median_sec']:.3f}s | mean {metrics['mean_sec']:.3f}s"
        )


if __name__ == "__main__":
    main()
