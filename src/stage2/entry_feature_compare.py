"""Compare cached features for entry-frame prediction against temporal priors."""

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
from torch import nn

from src.stage2.collision_feature_compare import CollisionFeatureModel
from src.stage2.entry_audit import (
    load_entry_labels,
    prior_metrics,
    temporal_prior_predictions,
)
from src.stage2.losses import make_temporal_loss


FEATURE_NAMES = ("global_center", "spatial_layer3", "spatial_layer4")


def _integer(value: Any, name: str) -> int:
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return int(number)


def _confidence(value: Any) -> str:
    if value is None or pd.isna(value) or not str(value).strip():
        return "UNKNOWN"
    return str(value).strip().upper()


def load_feature_samples(
    labels: pd.DataFrame,
    feature_dir: Path,
    prior_by_id: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config_path = feature_dir / "feature_config.json"
    feature_config = (
        json.loads(config_path.read_text("utf-8")) if config_path.is_file() else {}
    )
    samples = []
    dimensions: set[int] = set()
    channels: set[int] = set()
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
        target_frame = _integer(row.entry_frame, "entry_frame")
        matches = torch.where(frames == target_frame)[0]
        if len(matches) != 1:
            raise ValueError(
                f"entry_frame={target_frame} is not one cached frame for {row.ID}"
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
                "prior_frame": prior_by_id.get(str(row.ID)),
                "entry_confidence": _confidence(
                    getattr(row, "entry_confidence", None)
                ),
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


def entry_metrics(predictions: pd.DataFrame) -> dict[str, float | int]:
    errors = predictions["abs_error_sec"].astype(float)
    model_correct = predictions["model_correct_03"].astype(bool)
    prior_correct = predictions["prior_correct_03"].astype(bool)
    prior_wrong = ~prior_correct
    result: dict[str, float | int] = {
        "exact": float((predictions["error_frames"] == 0).mean()),
        "acc_01": float((errors <= 0.1 + 1e-9).mean()),
        "acc_02": float((errors <= 0.2 + 1e-9).mean()),
        "acc_03": float(model_correct.mean()),
        "acc_05": float((errors <= 0.5 + 1e-9).mean()),
        "median_sec": float(errors.median()),
        "mean_sec": float(errors.mean()),
        "prior_acc_03": float(prior_correct.mean()),
        "both_correct": int((model_correct & prior_correct).sum()),
        "model_only": int((model_correct & ~prior_correct).sum()),
        "prior_only": int((~model_correct & prior_correct).sum()),
        "both_wrong": int((~model_correct & ~prior_correct).sum()),
        "model_acc_on_prior_wrong": (
            float(model_correct[prior_wrong].mean()) if prior_wrong.any() else float("nan")
        ),
        "prior_correct_retention": (
            float(model_correct[prior_correct].mean())
            if prior_correct.any()
            else float("nan")
        ),
    }
    for confidence in ("HIGH", "MEDIUM", "LOW", "UNKNOWN"):
        selected = predictions["entry_confidence"].eq(confidence)
        if selected.any():
            result[f"{confidence.lower()}_count"] = int(selected.sum())
            result[f"{confidence.lower()}_acc_03"] = float(
                model_correct[selected].mean()
            )
    return result


@torch.inference_mode()
def evaluate(
    model: CollisionFeatureModel,
    samples: list[dict[str, Any]],
    loss_fn: nn.Module,
    device: torch.device,
    amp: bool,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    model.eval()
    losses = []
    rows = []
    for sample in samples:
        if sample["prior_frame"] is None:
            raise ValueError(f"Validation prior is missing for {sample['ID']}")
        inputs = sample["x"].unsqueeze(0).to(device=device, dtype=torch.float32)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp
        ):
            logits = model(inputs)
            loss = loss_fn(logits, _loss_sample(sample))
        predicted_index = int(logits.argmax(dim=1).item())
        predicted_frame = int(sample["frames"][predicted_index])
        error_frames = predicted_frame - sample["target_frame"]
        prior_error_frames = sample["prior_frame"] - sample["target_frame"]
        error_sec = abs(error_frames) / sample["fps"]
        prior_error_sec = abs(prior_error_frames) / sample["fps"]
        model_correct = error_sec <= 0.3 + 1e-9
        prior_correct = prior_error_sec <= 0.3 + 1e-9
        if model_correct and prior_correct:
            comparison = "both_correct"
        elif model_correct:
            comparison = "model_only"
        elif prior_correct:
            comparison = "prior_only"
        else:
            comparison = "both_wrong"
        losses.append(float(loss))
        rows.append(
            {
                "ID": sample["ID"],
                "target_frame": sample["target_frame"],
                "predicted_frame": predicted_frame,
                "prior_frame": sample["prior_frame"],
                "fps": sample["fps"],
                "error_frames": error_frames,
                "abs_error_sec": error_sec,
                "prior_error_frames": prior_error_frames,
                "prior_abs_error_sec": prior_error_sec,
                "model_correct_03": model_correct,
                "prior_correct_03": prior_correct,
                "comparison": comparison,
                "entry_confidence": sample["entry_confidence"],
            }
        )
    predictions = pd.DataFrame(rows)
    return {"loss": float(np.mean(losses)), **entry_metrics(predictions)}, predictions


def _reset_seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


def _checkpoint(
    model: nn.Module,
    model_config: dict[str, Any],
    feature_name: str,
    feature_config: dict[str, Any],
    loss_config: dict[str, Any],
    epoch: int,
    metrics: dict[str, float | int],
    seed: int,
) -> dict[str, Any]:
    return {
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "model_config": model_config,
        "feature_name": feature_name,
        "feature_config": feature_config,
        "loss_config": loss_config,
        "epoch": epoch,
        "metrics": metrics,
        "seed": seed,
        "task": "entry_frame_feature_compare",
    }


def train_one_feature(
    *,
    feature_name: str,
    feature_dir: Path,
    labels: pd.DataFrame,
    prior_by_id: dict[str, int],
    run_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    samples, feature_info = load_feature_samples(labels, feature_dir, prior_by_id)
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
        "delta_mode": args.delta_mode,
        "spatial_coordinates": args.spatial_coordinates,
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
        "early_stopping": {
            "metric": "validation loss",
            "patience": args.early_patience,
            "min_delta": args.early_min_delta,
        },
        "prior": "train median relative entry position",
        "train_ids": [sample["ID"] for sample in train_samples],
        "val_ids": [sample["ID"] for sample in val_samples],
        "seed": args.seed,
    }
    (experiment_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("=" * 88)
    print(
        f"Feature: {feature_name} | kind={feature_info['feature_kind']} | "
        f"channels={feature_info['input_channels']} | delta={args.delta_mode} | "
        f"spatial coordinates={args.spatial_coordinates}"
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
                device_type=device.type, dtype=torch.float16, enabled=amp
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
        checkpoint = _checkpoint(
            model,
            model_config,
            feature_name,
            feature_info["feature_config"],
            loss_fn.config,
            epoch,
            metrics,
            args.seed,
        )
        saved = []
        if accuracy_improved:
            best_acc = float(metrics["acc_03"])
            best_acc_tiebreak_loss = float(metrics["loss"])
            torch.save(checkpoint, best_path)
            torch.save(checkpoint, best_accuracy_path)
            predictions.to_csv(
                experiment_dir / "val_predictions.csv",
                index=False,
                encoding="utf-8-sig",
            )
            saved.append("best accuracy")
        if loss_improved:
            best_val_loss = float(metrics["loss"])
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
            f"@0.3 {metrics['acc_03']:.1%} | prior {metrics['prior_acc_03']:.1%} | "
            f"model-only {metrics['model_only']} | prior-only {metrics['prior_only']} | "
            f"median {metrics['median_sec']:.3f}s | {status}"
        )
        if next_lr < lr:
            print(f"{feature_name} | next LR: {next_lr:.2e}")
        if bad_epochs >= args.early_patience:
            print(f"{feature_name} | Early stopping on validation loss")
            break

    best = torch.load(best_path, map_location="cpu", weights_only=True)
    best_loss = torch.load(best_loss_path, map_location="cpu", weights_only=True)
    result = {
        "feature": feature_name,
        "feature_kind": feature_info["feature_kind"],
        "input_channels": feature_info["input_channels"],
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "best_epoch": best["epoch"],
        **best["metrics"],
        "checkpoint": str(best_path),
        "best_accuracy_checkpoint": str(best_accuracy_path),
        "best_loss_epoch": best_loss["epoch"],
        "best_val_loss": best_loss["metrics"]["loss"],
        "best_loss_acc_03": best_loss["metrics"]["acc_03"],
        "best_loss_checkpoint": str(best_loss_path),
    }
    print(f"{feature_name} | Best accuracy epoch: {best['epoch']}")
    print(f"{feature_name} | Best accuracy metrics: {best['metrics']}")
    print(f"{feature_name} | Best loss epoch: {best_loss['epoch']}")
    print(f"{feature_name} | Best loss metrics: {best_loss['metrics']}")
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
    parser.add_argument("--name", default="entry_feature_compare_v1")
    parser.add_argument(
        "--features", nargs="+", choices=FEATURE_NAMES, default=list(FEATURE_NAMES)
    )
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument("--projection-size", type=int, default=128)
    parser.add_argument("--temporal-input-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument(
        "--delta-mode",
        choices=("none", "concat"),
        default="none",
        help="concat appends signed feature[t]-feature[t-1] along channels",
    )
    parser.add_argument(
        "--spatial-coordinates",
        action="store_true",
        help=(
            "append attention mean x/y, spread and half-screen masses before "
            "the temporal model; spatial feature banks only"
        ),
    )
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
    labels = load_entry_labels(args.labels.expanduser().resolve())
    if args.expected_count is not None and len(labels) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} labels, got {len(labels)}")

    prior_predictions, priors = temporal_prior_predictions(labels)
    prior_table = prior_metrics(prior_predictions)
    selected_prior = prior_predictions[
        prior_predictions["method"] == "median_ratio"
    ].copy()
    prior_by_id = dict(
        zip(selected_prior["ID"], selected_prior["predicted_frame"].astype(int))
    )
    selected_prior_metrics = prior_table[
        prior_table["method"] == "median_ratio"
    ].iloc[0]
    print(f"Labels: {len(labels)} | Split: {labels['split'].value_counts().to_dict()}")
    print(
        "Content-free prior: median_ratio | "
        f"@0.3 {selected_prior_metrics['acc_03']:.1%} | "
        f"median {selected_prior_metrics['median_sec']:.3f}s"
    )
    print(f"Features: {args.features} | Delta: {args.delta_mode} | Loss: gaussian_ce")
    print(f"Spatial coordinates: {args.spatial_coordinates}")

    output_root = args.output_root.expanduser().resolve()
    feature_root = args.feature_root.expanduser().resolve()
    run_dir = output_root / (
        args.name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    labels.to_csv(run_dir / "labels_split.csv", index=False, encoding="utf-8-sig")
    prior_predictions.to_csv(
        run_dir / "temporal_prior_predictions.csv", index=False, encoding="utf-8-sig"
    )
    prior_table.to_csv(
        run_dir / "temporal_prior_metrics.csv", index=False, encoding="utf-8-sig"
    )
    (run_dir / "temporal_prior_config.json").write_text(
        json.dumps(priors, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

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
            prior_by_id=prior_by_id,
            run_dir=run_dir,
            args=args,
            device=device,
        )
        results.append(result)
        pd.DataFrame(results).to_csv(
            run_dir / "comparison.csv", index=False, encoding="utf-8-sig"
        )

    comparison = pd.DataFrame(results).sort_values(
        ["acc_03", "model_only", "prior_only", "median_sec", "loss"],
        ascending=[False, False, True, True, True],
    )
    comparison.to_csv(
        run_dir / "comparison_ranked.csv", index=False, encoding="utf-8-sig"
    )
    print("=" * 88)
    print("Entry feature comparison")
    print(
        comparison[
            [
                "feature",
                "best_epoch",
                "acc_03",
                "prior_acc_03",
                "model_only",
                "prior_only",
                "both_correct",
                "both_wrong",
                "median_sec",
                "mean_sec",
                "loss",
            ]
        ].to_string(index=False)
    )
    print(f"Run: {run_dir}")
    print(f"Ranked summary: {run_dir / 'comparison_ranked.csv'}")


if __name__ == "__main__":
    main()
