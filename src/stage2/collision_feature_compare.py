"""Compare global, layer3 and layer4 features for collision-frame prediction."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from src.stage2.collision_aihub_train import collision_metrics, load_collision_labels
from src.stage2.direct_model import SpatialAttentionPool
from src.stage2.losses import make_temporal_loss


FEATURE_NAMES = ("global_center", "spatial_layer3", "spatial_layer4")


class CollisionFeatureModel(nn.Module):
    """Use the same temporal model after either vector or spatial projection."""

    def __init__(
        self,
        *,
        feature_kind: str,
        input_channels: int,
        projection_size: int = 128,
        temporal_input_size: int = 256,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if feature_kind not in {"global", "spatial"}:
            raise ValueError("feature_kind must be global or spatial")
        self.feature_kind = feature_kind
        if feature_kind == "global":
            self.project = nn.Sequential(
                nn.Linear(input_channels, temporal_input_size),
                nn.LayerNorm(temporal_input_size),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.project = SpatialAttentionPool(
                input_channels=input_channels,
                projection_size=projection_size,
                output_size=temporal_input_size,
                dropout=dropout,
            )
        self.temporal = nn.GRU(
            temporal_input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.collision_head = nn.Linear(hidden_size * 2, 1)

    def forward(self, features: Tensor) -> Tensor:
        if self.feature_kind == "global":
            if features.ndim != 3:
                raise ValueError("Global features must have shape [B,T,C]")
            vectors = self.project(features)
        else:
            if features.ndim != 5:
                raise ValueError("Spatial features must have shape [B,T,C,H,W]")
            vectors, _ = self.project(features)
        hidden, _ = self.temporal(vectors)
        return self.collision_head(self.dropout(hidden)).squeeze(-1)


def _integer(value: Any, name: str) -> int:
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return int(number)


def load_feature_samples(
    labels: pd.DataFrame,
    feature_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config_path = feature_dir / "feature_config.json"
    feature_config = (
        json.loads(config_path.read_text("utf-8")) if config_path.is_file() else {}
    )
    samples = []
    dimensions = set()
    channels = set()
    for row in labels.itertuples(index=False):
        path = feature_dir / f"{row.ID}.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        cached = torch.load(path, map_location="cpu", weights_only=True)
        features = cached["features"]
        frames = cached["frame_numbers"].to(torch.int64)
        fps = float(cached["fps"])
        if features.ndim not in {2, 4}:
            raise ValueError(f"Features must be [T,C] or [T,C,H,W]: {path}")
        if len(features) != len(frames) or len(features) == 0:
            raise ValueError(f"Feature/frame length mismatch: {path}")
        if not torch.isfinite(features).all():
            raise ValueError(f"Non-finite features: {path}")
        if not (frames[1:] > frames[:-1]).all():
            raise ValueError(f"Frame numbers must increase: {path}")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid FPS: {path}")
        target_frame = _integer(row.collision_frame, "collision_frame")
        matches = torch.where(frames == target_frame)[0]
        if len(matches) != 1:
            raise ValueError(
                f"collision_frame={target_frame} is not one cached frame for {row.ID}"
            )
        dimensions.add(features.ndim)
        channels.add(int(features.shape[1]))
        samples.append(
            {
                "ID": str(row.ID),
                "split": str(row.split),
                "x": features,
                "frames": frames,
                "fps": fps,
                "target_frame": target_frame,
                "target_index": int(matches.item()),
            }
        )
    if len(dimensions) != 1 or len(channels) != 1:
        raise ValueError(
            f"Inconsistent feature cache: dimensions={dimensions}, channels={channels}"
        )
    feature_kind = "global" if next(iter(dimensions)) == 2 else "spatial"
    return samples, {
        "feature_kind": feature_kind,
        "input_channels": next(iter(channels)),
        "feature_config": feature_config,
    }


def _loss_sample(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "frames": sample["frames"],
        "fps": sample["fps"],
        "target_frame": sample["target_frame"],
        "target_index": sample["target_index"],
    }


@torch.inference_mode()
def evaluate(
    model: CollisionFeatureModel,
    samples: list[dict[str, Any]],
    loss_fn: nn.Module,
    device: torch.device,
    amp: bool,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    losses = []
    rows = []
    for sample in samples:
        inputs = sample["x"].unsqueeze(0).to(device=device, dtype=torch.float32)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp,
        ):
            logits = model(inputs)
            loss = loss_fn(logits, _loss_sample(sample))
        predicted_index = int(logits.argmax(dim=1).item())
        predicted_frame = int(sample["frames"][predicted_index])
        error_frames = predicted_frame - sample["target_frame"]
        losses.append(float(loss))
        rows.append(
            {
                "ID": sample["ID"],
                "target_frame": sample["target_frame"],
                "predicted_frame": predicted_frame,
                "fps": sample["fps"],
                "error_frames": error_frames,
                "abs_error_sec": abs(error_frames) / sample["fps"],
            }
        )
    predictions = pd.DataFrame(rows)
    return {"loss": float(np.mean(losses)), **collision_metrics(predictions)}, predictions


def _reset_seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


def train_one_feature(
    *,
    feature_name: str,
    feature_dir: Path,
    labels: pd.DataFrame,
    run_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    samples, feature_info = load_feature_samples(labels, feature_dir)
    train_samples = [sample for sample in samples if sample["split"] == "train"]
    val_samples = [sample for sample in samples if sample["split"] == "val"]
    if not train_samples or not val_samples:
        raise ValueError("Both train and val samples are required")
    rng = _reset_seed(args.seed)
    amp = not args.no_amp and device.type == "cuda"
    model_config = {
        "feature_kind": feature_info["feature_kind"],
        "input_channels": feature_info["input_channels"],
        "projection_size": args.projection_size,
        "temporal_input_size": args.temporal_input_size,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
    }
    model = CollisionFeatureModel(**model_config).to(device)
    loss_fn = make_temporal_loss("gaussian_ce", sigma_sec=args.sigma_sec)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    experiment_dir = run_dir / feature_name
    experiment_dir.mkdir(parents=True, exist_ok=False)
    best_path = experiment_dir / "best_model.pt"
    best_accuracy_path = experiment_dir / "best_accuracy_model.pt"
    best_loss_path = experiment_dir / "best_loss_model.pt"
    config = {
        "feature_name": feature_name,
        "feature_dir": str(feature_dir),
        "feature_config": feature_info["feature_config"],
        "model_config": model_config,
        "loss": loss_fn.config,
        "optimizer": {
            "name": "AdamW",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        "scheduler": {
            "name": "ReduceLROnPlateau",
            "factor": 0.5,
            "patience": 2,
            "min_lr": 1e-6,
        },
        "checkpoint_selection": {
            "best_model.pt": "highest Accuracy@0.3s, then lowest validation loss",
            "best_accuracy_model.pt": "same checkpoint as best_model.pt",
            "best_loss_model.pt": "lowest validation loss",
            "early_stopping": "validation loss",
            "early_min_delta": args.early_min_delta,
            "early_patience": args.early_patience,
        },
        "train_ids": [sample["ID"] for sample in train_samples],
        "val_ids": [sample["ID"] for sample in val_samples],
        "seed": args.seed,
    }
    (experiment_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("=" * 88)
    print(
        f"Feature: {feature_name} | kind={feature_info['feature_kind']} | "
        f"channels={feature_info['input_channels']}"
    )
    print(
        f"Train: {len(train_samples)} | Val: {len(val_samples)} | "
        f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    best_acc = -1.0
    best_acc_tiebreak_loss = float("inf")
    best_val_loss = float("inf")
    bad_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        lr = optimizer.param_groups[0]["lr"]
        for sample_index in rng.permutation(len(train_samples)):
            sample = train_samples[int(sample_index)]
            inputs = sample["x"].unsqueeze(0).to(
                device=device, dtype=torch.float32
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp,
            ):
                logits = model(inputs)
                loss = loss_fn(logits, _loss_sample(sample))
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss for {sample['ID']}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach()))

        metrics, predictions = evaluate(model, val_samples, loss_fn, device, amp)
        scheduler.step(metrics["loss"])
        next_lr = optimizer.param_groups[0]["lr"]

        accuracy_improved = metrics["acc_03"] > best_acc + 1e-12 or (
            abs(metrics["acc_03"] - best_acc) <= 1e-12
            and metrics["loss"] < best_acc_tiebreak_loss - 1e-6
        )
        loss_improved = metrics["loss"] < best_val_loss - args.early_min_delta
        checkpoint = {
            "model_state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "model_config": model_config,
            "feature_name": feature_name,
            "feature_config": feature_info["feature_config"],
            "loss_config": loss_fn.config,
            "epoch": epoch,
            "metrics": metrics,
            "seed": args.seed,
            "task": "collision_frame_feature_compare",
        }
        saved = []
        if accuracy_improved:
            best_acc = metrics["acc_03"]
            best_acc_tiebreak_loss = metrics["loss"]
            torch.save(checkpoint, best_path)
            torch.save(checkpoint, best_accuracy_path)
            predictions.to_csv(
                experiment_dir / "val_predictions.csv",
                index=False,
                encoding="utf-8-sig",
            )
            saved.append("best accuracy")

        if loss_improved:
            best_val_loss = metrics["loss"]
            bad_epochs = 0
            torch.save(checkpoint, best_loss_path)
            predictions.to_csv(
                experiment_dir / "val_predictions_best_loss.csv",
                index=False,
                encoding="utf-8-sig",
            )
            saved.append("best loss")
        else:
            bad_epochs += 1
        status = ", ".join(saved) if saved else "no checkpoint update"
        status += f" | loss patience {bad_epochs}/{args.early_patience}"

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "lr": lr,
                "next_lr": next_lr,
                "bad_epochs": bad_epochs,
                "accuracy_improved": accuracy_improved,
                "loss_improved": loss_improved,
                **metrics,
            }
        )
        pd.DataFrame(history).to_csv(experiment_dir / "history.csv", index=False)
        print(
            f"{feature_name} | Epoch {epoch:02d} | LR {lr:.2e} | "
            f"train {history[-1]['train_loss']:.4f} | val {metrics['loss']:.4f} | "
            f"@0.3 {metrics['acc_03']:.1%} | exact {metrics['exact']:.1%} | "
            f"median {metrics['median_sec']:.3f}s | {status}"
        )
        if next_lr < lr:
            print(f"{feature_name} | next LR: {next_lr:.2e}")
        if bad_epochs >= args.early_patience:
            print(f"{feature_name} | Early stopping on validation loss")
            break

    best = torch.load(best_path, map_location="cpu", weights_only=True)
    best_loss_checkpoint = torch.load(
        best_loss_path, map_location="cpu", weights_only=True
    )
    result = {
        "feature": feature_name,
        "feature_kind": feature_info["feature_kind"],
        "input_channels": feature_info["input_channels"],
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "best_epoch": best["epoch"],
        **best["metrics"],
        "checkpoint": str(best_path),
        "best_accuracy_checkpoint": str(best_accuracy_path),
        "best_loss_epoch": best_loss_checkpoint["epoch"],
        "best_val_loss": best_loss_checkpoint["metrics"]["loss"],
        "best_loss_acc_03": best_loss_checkpoint["metrics"]["acc_03"],
        "best_loss_checkpoint": str(best_loss_path),
    }
    print(f"{feature_name} | Best accuracy epoch: {best['epoch']}")
    print(f"{feature_name} | Best accuracy metrics: {best['metrics']}")
    print(f"{feature_name} | Best loss epoch: {best_loss_checkpoint['epoch']}")
    print(f"{feature_name} | Best loss metrics: {best_loss_checkpoint['metrics']}")
    del model, optimizer, scheduler, scaler, samples, train_samples, val_samples
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--name", default="collision_feature_compare_v1")
    parser.add_argument(
        "--features",
        nargs="+",
        choices=FEATURE_NAMES,
        default=list(FEATURE_NAMES),
    )
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument("--projection-size", type=int, default=128)
    parser.add_argument("--temporal-input-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--sigma-sec", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early-patience", type=int, default=10)
    parser.add_argument("--early-min-delta", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.early_patience < 1:
        raise ValueError("epochs and early-patience must be positive")
    if args.early_min_delta < 0:
        raise ValueError("early-min-delta cannot be negative")
    labels_path = args.labels.expanduser().resolve()
    feature_root = args.feature_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    labels = load_collision_labels(labels_path)
    if args.expected_count is not None and len(labels) != args.expected_count:
        raise ValueError(
            f"Expected {args.expected_count} usable collision labels, got {len(labels)}"
        )
    split_counts = labels["split"].value_counts().to_dict()
    print(f"Labels: {len(labels)} | Split: {split_counts}")
    print(f"Features: {args.features}")
    print("Delta: disabled | Loss: gaussian_ce")

    run_dir = output_root / (
        args.name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    labels.to_csv(run_dir / "labels_split.csv", index=False, encoding="utf-8-sig")
    results = []
    device = torch.device(args.device)
    for feature_name in args.features:
        feature_dir = feature_root / feature_name
        if not feature_dir.is_dir():
            raise FileNotFoundError(feature_dir)
        result = train_one_feature(
            feature_name=feature_name,
            feature_dir=feature_dir,
            labels=labels,
            run_dir=run_dir,
            args=args,
            device=device,
        )
        results.append(result)
        pd.DataFrame(results).to_csv(
            run_dir / "comparison.csv", index=False, encoding="utf-8-sig"
        )

    comparison = pd.DataFrame(results).sort_values(
        ["acc_03", "median_sec", "loss"],
        ascending=[False, True, True],
    )
    comparison.to_csv(
        run_dir / "comparison_ranked.csv", index=False, encoding="utf-8-sig"
    )
    print("=" * 88)
    print("Collision feature comparison")
    print(
        comparison[
            [
                "feature",
                "best_epoch",
                "acc_03",
                "exact",
                "median_sec",
                "loss",
                "best_loss_epoch",
                "best_val_loss",
                "best_loss_acc_03",
                "parameters",
            ]
        ].to_string(index=False)
    )
    print(f"Run: {run_dir}")
    print(f"Ranked summary: {run_dir / 'comparison_ranked.csv'}")


if __name__ == "__main__":
    main()
