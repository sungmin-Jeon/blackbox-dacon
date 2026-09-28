"""Train entry-frame, entry-side and evasion-space heads on cached spatial maps."""

from __future__ import annotations

import argparse
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

from src.stage2.direct_model import Stage2DirectSpatial
from src.stage2.losses import make_temporal_loss


SIDE_TO_INDEX = {"LEFT": 0, "RIGHT": 1}


def _present(value: Any) -> bool:
    return value is not None and not pd.isna(value) and str(value).strip() != ""


def _integer(value: Any, name: str) -> int:
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return int(number)


def _frame_index(frames: Tensor, value: Any, name: str) -> int:
    target = _integer(value, name)
    matches = torch.where(frames.to(torch.int64) == target)[0]
    if len(matches) != 1:
        raise ValueError(f"{name}={target} is not exactly one cached frame")
    return int(matches.item())


def _load_labels(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(path, dtype={"ID": str})
    required = {"ID", "split", "entry_side", "evasion_space"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Missing label columns: {sorted(missing)}")
    if "entry_frame" not in labels and "entry_best" in labels:
        labels["entry_frame"] = labels["entry_best"]
    if "collision_frame" not in labels and "collision_best" in labels:
        labels["collision_frame"] = labels["collision_best"]
    if "entry_frame" not in labels or "collision_frame" not in labels:
        raise ValueError("Labels require entry_frame and collision_frame (or *_best aliases)")
    if "status" in labels:
        labels = labels[labels["status"].fillna("").str.upper().eq("KEEP")]
    labels = labels.copy()
    labels["split"] = labels["split"].astype(str).str.lower()
    labels = labels[labels["split"].isin({"train", "val"})]
    if labels["ID"].duplicated().any():
        duplicates = labels.loc[labels["ID"].duplicated(False), "ID"].tolist()
        raise ValueError(f"Duplicate IDs in labels: {duplicates[:10]}")
    if not {"train", "val"}.issubset(set(labels["split"])):
        raise ValueError("Both train and val splits are required")
    return labels.reset_index(drop=True)


def _load_sample(row: Any, feature_dir: Path) -> dict[str, Any]:
    path = feature_dir / f"{row.ID}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    cached = torch.load(path, map_location="cpu", weights_only=True)
    maps = cached["features"]
    frames = cached["frame_numbers"].to(torch.int64)
    if maps.ndim != 4 or len(maps) != len(frames) or not torch.isfinite(maps).all():
        raise ValueError(f"Invalid feature cache: {path}")
    if not (frames[1:] > frames[:-1]).all():
        raise ValueError(f"Frame numbers must increase: {path}")
    fps = float(cached["fps"])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid FPS in {path}")

    entry_index = _frame_index(frames, row.entry_frame, "entry_frame") if _present(row.entry_frame) else None
    collision_index = (
        _frame_index(frames, row.collision_frame, "collision_frame")
        if _present(row.collision_frame) else None
    )
    side = None
    if _present(row.entry_side):
        side_name = str(row.entry_side).strip().upper()
        if side_name not in SIDE_TO_INDEX:
            raise ValueError(f"Invalid entry_side for {row.ID}: {row.entry_side!r}")
        side = SIDE_TO_INDEX[side_name]
    evasion = None
    if _present(row.evasion_space):
        evasion = _integer(row.evasion_space, "evasion_space")
        if evasion not in (0, 1):
            raise ValueError(f"Invalid evasion_space for {row.ID}: {evasion}")
    if evasion is not None and collision_index is None:
        raise ValueError(f"Evasion label requires collision_frame: {row.ID}")
    if entry_index is None and side is None and evasion is None:
        raise ValueError(f"No usable direct labels: {row.ID}")
    return {
        "ID": str(row.ID),
        "split": str(row.split),
        "maps": maps,
        "frames": frames,
        "fps": fps,
        "entry_index": entry_index,
        "entry_frame": int(frames[entry_index]) if entry_index is not None else None,
        "collision_index": collision_index,
        "side": side,
        "evasion": evasion,
    }


def _macro_f1(targets: list[int], predictions: list[int]) -> float:
    if not targets:
        return float("nan")
    values = []
    for label in (0, 1):
        tp = sum(t == label and p == label for t, p in zip(targets, predictions))
        fp = sum(t != label and p == label for t, p in zip(targets, predictions))
        fn = sum(t == label and p != label for t, p in zip(targets, predictions))
        denominator = 2 * tp + fp + fn
        values.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(values))


def _class_weights(samples: list[dict[str, Any]], key: str, device: torch.device) -> Tensor:
    counts = torch.tensor([sum(s[key] == label for s in samples) for label in (0, 1)], dtype=torch.float32)
    if (counts == 0).any():
        raise ValueError(f"Cannot balance {key}; train split lacks one class: {counts.tolist()}")
    weights = counts.sum() / (2 * counts)
    return weights.to(device)


def _entry_loss_sample(sample: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "frames": sample["frames"],
        "target_index": sample["entry_index"],
        "target_frame": sample["entry_frame"],
        "fps": sample["fps"],
    }


def _context_index(sample: dict[str, Any], *, jitter: int, rng: np.random.Generator | None) -> int:
    if sample["collision_index"] is None:
        return len(sample["maps"]) - 1
    index = sample["collision_index"]
    if jitter and rng is not None:
        index += int(rng.integers(-jitter, jitter + 1))
    return min(max(index, 0), len(sample["maps"]) - 1)


def _combined_loss(
    outputs: dict[str, Tensor],
    sample: dict[str, Any],
    entry_loss_fn: nn.Module,
    side_loss_fn: nn.Module,
    evasion_loss_fn: nn.Module,
    weights: dict[str, float],
    device: torch.device,
) -> tuple[Tensor, dict[str, float]]:
    terms: list[tuple[str, Tensor, float]] = []
    if sample["entry_index"] is not None:
        terms.append(("entry", entry_loss_fn(outputs["entry_logits"], _entry_loss_sample(sample, device)), weights["entry"]))
    if sample["side"] is not None:
        target = torch.tensor([sample["side"]], device=device)
        terms.append(("side", side_loss_fn(outputs["side_logits"], target), weights["side"]))
    if sample["evasion"] is not None:
        target = torch.tensor([sample["evasion"]], device=device)
        terms.append(("evasion", evasion_loss_fn(outputs["evasion_logits"], target), weights["evasion"]))
    if not terms:
        raise ValueError(f"No active loss for {sample['ID']}")
    total_weight = sum(weight for _, _, weight in terms)
    total = sum(loss * weight for _, loss, weight in terms) / total_weight
    return total, {name: float(loss.detach()) for name, loss, _ in terms}


@torch.inference_mode()
def evaluate(
    model: Stage2DirectSpatial,
    samples: list[dict[str, Any]],
    entry_loss_fn: nn.Module,
    side_loss_fn: nn.Module,
    evasion_loss_fn: nn.Module,
    loss_weights: dict[str, float],
    device: torch.device,
    amp: bool,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    losses = []
    rows = []
    for sample in samples:
        maps = sample["maps"].unsqueeze(0).to(device=device, dtype=torch.float32)
        collision_index = _context_index(sample, jitter=0, rng=None)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            outputs = model(maps, torch.tensor([collision_index], device=device))
            loss, _ = _combined_loss(
                outputs, sample, entry_loss_fn, side_loss_fn, evasion_loss_fn, loss_weights, device
            )
        losses.append(float(loss))
        row: dict[str, Any] = {"ID": sample["ID"]}
        if sample["entry_index"] is not None:
            predicted_index = int(outputs["entry_logits"].argmax(dim=1).item())
            predicted_frame = int(sample["frames"][predicted_index])
            error_seconds = abs(
                predicted_frame - int(sample["frames"][sample["entry_index"]])
            ) / sample["fps"]
            row.update(
                target_entry_frame=sample["entry_frame"],
                predicted_entry_frame=predicted_frame,
                entry_error_seconds=error_seconds,
                entry_correct_03=error_seconds <= 0.3 + 1e-9,
                entry_exact=predicted_index == sample["entry_index"],
            )
        if sample["side"] is not None:
            row.update(
                target_side=sample["side"],
                predicted_side=int(outputs["side_logits"].argmax(1).item()),
            )
        if sample["evasion"] is not None:
            row.update(
                target_evasion=sample["evasion"],
                predicted_evasion=int(outputs["evasion_logits"].argmax(1).item()),
            )
        rows.append(row)

    predictions = pd.DataFrame(rows)
    entry_rows = predictions.dropna(subset=["entry_correct_03"]) if "entry_correct_03" in predictions else predictions.iloc[0:0]
    side_rows = predictions.dropna(subset=["target_side"]) if "target_side" in predictions else predictions.iloc[0:0]
    evasion_rows = predictions.dropna(subset=["target_evasion"]) if "target_evasion" in predictions else predictions.iloc[0:0]
    entry_acc = float(entry_rows["entry_correct_03"].mean()) if len(entry_rows) else float("nan")
    side_f1 = _macro_f1(side_rows["target_side"].astype(int).tolist(), side_rows["predicted_side"].astype(int).tolist())
    evasion_f1 = _macro_f1(
        evasion_rows["target_evasion"].astype(int).tolist(),
        evasion_rows["predicted_evasion"].astype(int).tolist(),
    )
    available_weight = sum(
        weight for value, weight in ((entry_acc, 0.35), (side_f1, 0.15), (evasion_f1, 0.15))
        if math.isfinite(value)
    )
    direct_score = sum(
        value * weight for value, weight in ((entry_acc, 0.35), (side_f1, 0.15), (evasion_f1, 0.15))
        if math.isfinite(value)
    ) / available_weight
    metrics = {
        "loss": float(np.mean(losses)),
        "entry_acc_03": entry_acc,
        "entry_exact": float(entry_rows["entry_exact"].mean()) if len(entry_rows) else float("nan"),
        "entry_median_sec": float(entry_rows["entry_error_seconds"].median()) if len(entry_rows) else float("nan"),
        "side_macro_f1": side_f1,
        "evasion_macro_f1": evasion_f1,
        "direct_score_normalized": direct_score,
        "evasion_context": "ground_truth_collision_frame",
    }
    return metrics, predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--name", default="direct_spatial_v1")
    parser.add_argument("--entry-loss", choices=("hard_ce", "gaussian_ce", "window_ce", "window_nll"), default="gaussian_ce")
    parser.add_argument("--sigma-sec", type=float, default=0.1)
    parser.add_argument("--radius-sec", type=float, default=0.3)
    parser.add_argument("--projection-size", type=int, default=128)
    parser.add_argument("--temporal-input-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--collision-window", type=int, default=2)
    parser.add_argument("--collision-jitter", type=int, default=4)
    parser.add_argument("--entry-weight", type=float, default=1.0)
    parser.add_argument("--side-weight", type=float, default=0.5)
    parser.add_argument("--evasion-weight", type=float, default=0.5)
    parser.add_argument("--class-balance", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early-patience", type=int, default=7)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.collision_jitter < 0 or args.epochs < 1 or args.early_patience < 1:
        raise ValueError("collision-jitter must be nonnegative; epochs/patience must be positive")
    labels_path = args.labels.expanduser().resolve()
    feature_dir = args.feature_dir.expanduser().resolve()
    labels = _load_labels(labels_path)
    samples = [_load_sample(row, feature_dir) for row in labels.itertuples(index=False)]
    train_samples = [sample for sample in samples if sample["split"] == "train"]
    val_samples = [sample for sample in samples if sample["split"] == "val"]
    if {s["ID"] for s in train_samples} & {s["ID"] for s in val_samples}:
        raise ValueError("Train/validation IDs overlap")
    channel_counts = {int(s["maps"].shape[1]) for s in samples}
    if len(channel_counts) != 1:
        raise ValueError(f"Feature channel count is inconsistent: {channel_counts}")
    input_channels = samples[0]["maps"].shape[1]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    amp = not args.no_amp and device.type == "cuda"

    model_config = {
        "input_channels": input_channels,
        "projection_size": args.projection_size,
        "temporal_input_size": args.temporal_input_size,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "collision_window": args.collision_window,
        "entry_temperature": 1.0,
    }
    model = Stage2DirectSpatial(**model_config).to(device)
    entry_options: dict[str, Any] = {}
    if args.entry_loss == "gaussian_ce":
        entry_options["sigma_sec"] = args.sigma_sec
    elif args.entry_loss in {"window_ce", "window_nll"}:
        entry_options["radius_sec"] = args.radius_sec
    entry_loss_fn = make_temporal_loss(args.entry_loss, **entry_options)
    side_weight = _class_weights(train_samples, "side", device) if args.class_balance else None
    evasion_weight = _class_weights(train_samples, "evasion", device) if args.class_balance else None
    side_loss_fn = nn.CrossEntropyLoss(weight=side_weight)
    evasion_loss_fn = nn.CrossEntropyLoss(weight=evasion_weight)
    loss_weights = {"entry": args.entry_weight, "side": args.side_weight, "evasion": args.evasion_weight}
    if any(weight <= 0 for weight in loss_weights.values()):
        raise ValueError("All task loss weights must be positive")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    run_dir = args.output_root.expanduser().resolve() / (
        args.name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    best_path = run_dir / "best_model.pt"
    labels.to_csv(run_dir / "labels_split.csv", index=False, encoding="utf-8-sig")
    feature_config_path = feature_dir / "feature_config.json"
    feature_config = json.loads(feature_config_path.read_text("utf-8")) if feature_config_path.is_file() else {}
    experiment_config = {
        "model": model_config,
        "entry_loss": entry_loss_fn.config,
        "task_loss_weights": loss_weights,
        "class_balance": args.class_balance,
        "collision_jitter": args.collision_jitter,
        "optimizer": {"name": "AdamW", "lr": args.lr, "weight_decay": args.weight_decay},
        "scheduler": {"name": "ReduceLROnPlateau", "factor": 0.5, "patience": 2, "min_lr": 1e-6},
        "epochs": args.epochs,
        "early_patience": args.early_patience,
        "seed": args.seed,
        "labels": str(labels_path),
        "feature_dir": str(feature_dir),
        "feature_config": feature_config,
        "train_ids": [s["ID"] for s in train_samples],
        "val_ids": [s["ID"] for s in val_samples],
    }
    (run_dir / "config.json").write_text(
        json.dumps(experiment_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Device: {device} | AMP: {amp}")
    print(f"Train: {len(train_samples)} | Val: {len(val_samples)}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Run: {run_dir}")

    best_score, best_loss, bad_epochs = -float("inf"), float("inf"), 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        lr = optimizer.param_groups[0]["lr"]
        for sample_index in rng.permutation(len(train_samples)):
            sample = train_samples[int(sample_index)]
            maps = sample["maps"].unsqueeze(0).to(device=device, dtype=torch.float32)
            collision_index = _context_index(
                sample, jitter=args.collision_jitter, rng=rng
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                outputs = model(maps, torch.tensor([collision_index], device=device))
                loss, _ = _combined_loss(
                    outputs, sample, entry_loss_fn, side_loss_fn, evasion_loss_fn, loss_weights, device
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss for {sample['ID']}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach()))

        metrics, predictions = evaluate(
            model, val_samples, entry_loss_fn, side_loss_fn, evasion_loss_fn,
            loss_weights, device, amp,
        )
        scheduler.step(metrics["loss"])
        score = metrics["direct_score_normalized"]
        improved = score > best_score + 1e-12 or (
            abs(score - best_score) <= 1e-12 and metrics["loss"] < best_loss - 1e-6
        )
        if improved:
            best_score, best_loss, bad_epochs = score, metrics["loss"], 0
            checkpoint = {
                "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "model_config": model_config,
                "feature_config": feature_config,
                "entry_loss_config": entry_loss_fn.config,
                "task_loss_weights": loss_weights,
                "label_mapping": {"entry_side": SIDE_TO_INDEX, "evasion_space": {"0": 0, "1": 1}},
                "epoch": epoch,
                "metrics": metrics,
                "seed": args.seed,
                "task": "entry_frame_entry_side_evasion_space",
            }
            torch.save(checkpoint, best_path)
            predictions.to_csv(run_dir / "val_predictions.csv", index=False, encoding="utf-8-sig")
            status = "best saved"
        else:
            bad_epochs += 1
            status = f"no improvement {bad_epochs}/{args.early_patience}"

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "lr": lr,
            "next_lr": optimizer.param_groups[0]["lr"],
            "bad_epochs": bad_epochs,
            **metrics,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        print(
            f"Epoch {epoch:02d} | train {row['train_loss']:.4f} | val {metrics['loss']:.4f} | "
            f"entry@0.3 {metrics['entry_acc_03']:.1%} | side F1 {metrics['side_macro_f1']:.3f} | "
            f"evasion F1 {metrics['evasion_macro_f1']:.3f} | direct {score:.3f} | {status}"
        )
        if bad_epochs >= args.early_patience:
            print("Early stopping")
            break

    best = torch.load(best_path, map_location="cpu", weights_only=True)
    print(f"Best epoch: {best['epoch']}")
    print(f"Best metrics: {best['metrics']}")
    print(f"Checkpoint: {best_path}")
    print("Caution: validation evasion uses the annotated collision frame; submission uses the collision model prediction.")


if __name__ == "__main__":
    main()
