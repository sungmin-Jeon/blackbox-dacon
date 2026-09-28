"""Train a fresh collision-frame BiGRU using only directly labelled AIHub videos."""

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
from torchvision.models import ResNet18_Weights, resnet18
from tqdm.auto import tqdm

from src.stage2.collision_predict import (
    CollisionBiGRU,
    _video_index,
    extract_video_features,
)
from src.stage2.losses import make_temporal_loss


FEATURE_CONFIG = {
    "version": 1,
    "backbone": "resnet18_imagenet1k_v1",
    "backbone_output": "avgpool_512",
    "preprocess": "ResNet18_Weights.IMAGENET1K_V1.transforms",
    "frame_number_base": 1,
    "storage_dtype": "float32",
}


def _present(value: Any) -> bool:
    return value is not None and not pd.isna(value) and str(value).strip() != ""


def _integer(value: Any, name: str) -> int:
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return int(number)


def load_collision_labels(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(path, dtype={"ID": str})
    if "ID" not in labels:
        raise ValueError("Labels require an ID column")
    if "collision_frame" not in labels and "collision_best" in labels:
        labels["collision_frame"] = labels["collision_best"]
    required = {"ID", "split", "collision_frame"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Missing label columns: {sorted(missing)}")
    if "status" in labels:
        labels = labels[labels["status"].fillna("").str.upper().eq("KEEP")]
    labels = labels.copy()
    labels["split"] = labels["split"].astype(str).str.lower()
    labels = labels[labels["split"].isin({"train", "val"})]
    labels = labels[labels["collision_frame"].map(_present)].copy()
    labels["collision_frame"] = [
        _integer(value, "collision_frame") for value in labels["collision_frame"]
    ]
    if labels["ID"].duplicated().any():
        duplicates = labels.loc[labels["ID"].duplicated(False), "ID"].tolist()
        raise ValueError(f"Duplicate IDs in labels: {duplicates[:10]}")
    if not {"train", "val"}.issubset(set(labels["split"])):
        raise ValueError("Collision-labelled rows require both train and val splits")
    return labels.reset_index(drop=True)


def _write_feature_config(feature_dir: Path) -> None:
    path = feature_dir / "feature_config.json"
    if path.is_file():
        existing = json.loads(path.read_text("utf-8"))
        if existing != FEATURE_CONFIG:
            raise ValueError(
                "Existing collision feature cache uses different settings; "
                "choose another --feature-dir"
            )
    else:
        path.write_text(
            json.dumps(FEATURE_CONFIG, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def cache_collision_features(
    labels: pd.DataFrame,
    video_root: Path,
    feature_dir: Path,
    backbone_path: Path,
    device: torch.device,
    batch_size: int,
    force: bool = False,
) -> None:
    feature_dir.mkdir(parents=True, exist_ok=True)
    _write_feature_config(feature_dir)
    videos = _video_index(video_root)
    missing = [video_id for video_id in labels["ID"] if video_id not in videos]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} videos: {missing[:10]}")

    pending = [
        video_id
        for video_id in labels["ID"]
        if force or not (feature_dir / f"{video_id}.pt").is_file()
    ]
    if not pending:
        print(f"Feature cache: all {len(labels)} videos already exist")
        return

    backbone = resnet18(weights=None)
    backbone.load_state_dict(
        torch.load(backbone_path, map_location="cpu", weights_only=True)
    )
    backbone.fc = nn.Identity()
    backbone.to(device).eval().requires_grad_(False)
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()

    for video_id in tqdm(pending, desc="Caching AIHub collision features"):
        features, fps = extract_video_features(
            videos[video_id], backbone, transform, device, batch_size
        )
        frame_numbers = torch.arange(1, len(features) + 1, dtype=torch.int64)
        cache = {
            "ID": video_id,
            "features": features.to(torch.float32),
            "frame_numbers": frame_numbers,
            "fps": float(fps),
            "feature_config": FEATURE_CONFIG,
        }
        output = feature_dir / f"{video_id}.pt"
        temporary = output.with_suffix(".pt.tmp")
        torch.save(cache, temporary)
        temporary.replace(output)
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()


def load_collision_sample(row: Any, feature_dir: Path) -> dict[str, Any]:
    path = feature_dir / f"{row.ID}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    cached = torch.load(path, map_location="cpu", weights_only=True)
    features = cached["features"].to(torch.float32)
    frames = cached["frame_numbers"].to(torch.int64)
    fps = float(cached["fps"])
    if features.ndim != 2 or features.shape[1] != 512 or len(features) != len(frames):
        raise ValueError(f"Invalid features in {path}")
    if not torch.isfinite(features).all() or not (frames[1:] > frames[:-1]).all():
        raise ValueError(f"Invalid feature values or frame numbers in {path}")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid FPS in {path}")
    target_frame = _integer(row.collision_frame, "collision_frame")
    matches = torch.where(frames == target_frame)[0]
    if len(matches) != 1:
        raise ValueError(
            f"collision_frame={target_frame} is not one cached frame for {row.ID}"
        )
    return {
        "ID": str(row.ID),
        "split": str(row.split),
        "x": features,
        "frames": frames,
        "fps": fps,
        "target_frame": target_frame,
        "target_index": int(matches.item()),
    }


def _loss_sample(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "frames": sample["frames"],
        "fps": sample["fps"],
        "target_frame": sample["target_frame"],
        "target_index": sample["target_index"],
    }


def collision_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    errors = predictions["abs_error_sec"].astype(float)
    return {
        "exact": float((predictions["error_frames"] == 0).mean()),
        "acc_01": float((errors <= 0.1 + 1e-9).mean()),
        "acc_02": float((errors <= 0.2 + 1e-9).mean()),
        "acc_03": float((errors <= 0.3 + 1e-9).mean()),
        "acc_05": float((errors <= 0.5 + 1e-9).mean()),
        "median_sec": float(errors.median()),
        "mean_sec": float(errors.mean()),
    }


@torch.inference_mode()
def evaluate_collision(
    model: CollisionBiGRU,
    samples: list[dict[str, Any]],
    loss_fn: nn.Module,
    device: torch.device,
    amp: bool,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    losses = []
    rows = []
    for sample in samples:
        inputs = sample["x"].unsqueeze(0).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
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


def baseline_metrics_on_validation(
    prediction_path: Path | None,
    val_samples: list[dict[str, Any]],
) -> tuple[dict[str, float] | None, pd.DataFrame | None]:
    if prediction_path is None:
        return None, None
    predictions = pd.read_csv(prediction_path, dtype={"ID": str})
    required = {"ID", "collision_pred_frame"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Baseline predictions miss columns: {sorted(missing)}")
    if predictions["ID"].duplicated().any():
        raise ValueError("Baseline predictions contain duplicate IDs")
    predicted_by_id = predictions.set_index("ID")["collision_pred_frame"]
    rows = []
    for sample in val_samples:
        if sample["ID"] not in predicted_by_id.index:
            raise ValueError(f"Baseline prediction missing for {sample['ID']}")
        predicted_frame = _integer(
            predicted_by_id.loc[sample["ID"]], "collision_pred_frame"
        )
        error_frames = predicted_frame - sample["target_frame"]
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
    frame = pd.DataFrame(rows)
    return collision_metrics(frame), frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--name", default="collision_aihub_only_v1")
    parser.add_argument("--baseline-predictions", type=Path, default=None)
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--loss", choices=("hard_ce", "gaussian_ce", "window_ce", "window_nll"), default="gaussian_ce")
    parser.add_argument("--sigma-sec", type=float, default=0.1)
    parser.add_argument("--radius-sec", type=float, default=0.3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
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
    if args.feature_batch_size < 1 or args.epochs < 1 or args.early_patience < 1:
        raise ValueError("Batch size, epochs and patience must be positive")
    labels_path = args.labels.expanduser().resolve()
    video_root = args.video_root.expanduser().resolve()
    backbone_path = args.backbone.expanduser().resolve()
    feature_dir = args.feature_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    baseline_path = (
        args.baseline_predictions.expanduser().resolve()
        if args.baseline_predictions is not None
        else None
    )
    labels = load_collision_labels(labels_path)
    device = torch.device(args.device)
    cache_collision_features(
        labels,
        video_root,
        feature_dir,
        backbone_path,
        device,
        args.feature_batch_size,
        args.force_features,
    )
    samples = [
        load_collision_sample(row, feature_dir)
        for row in labels.itertuples(index=False)
    ]
    train_samples = [sample for sample in samples if sample["split"] == "train"]
    val_samples = [sample for sample in samples if sample["split"] == "val"]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    amp = not args.no_amp and device.type == "cuda"

    model_config = {
        "input_size": 512,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
    }
    model = CollisionBiGRU(**model_config).to(device)
    loss_options: dict[str, float] = {}
    if args.loss == "gaussian_ce":
        loss_options["sigma_sec"] = args.sigma_sec
    elif args.loss in {"window_ce", "window_nll"}:
        loss_options["radius_sec"] = args.radius_sec
    loss_fn = make_temporal_loss(args.loss, **loss_options)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    run_dir = output_root / (
        args.name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    best_path = run_dir / "best_model.pt"
    labels.to_csv(run_dir / "labels_split.csv", index=False, encoding="utf-8-sig")
    baseline_metrics, baseline_rows = baseline_metrics_on_validation(
        baseline_path, val_samples
    )
    if baseline_rows is not None:
        baseline_rows.to_csv(
            run_dir / "baseline_val_predictions.csv", index=False, encoding="utf-8-sig"
        )
    config = {
        "model": model_config,
        "loss": loss_fn.config,
        "optimizer": {"name": "AdamW", "lr": args.lr, "weight_decay": args.weight_decay},
        "scheduler": {"name": "ReduceLROnPlateau", "factor": 0.5, "patience": 2, "min_lr": 1e-6},
        "labels": str(labels_path),
        "video_root": str(video_root),
        "feature_dir": str(feature_dir),
        "feature_config": FEATURE_CONFIG,
        "baseline_predictions": str(baseline_path) if baseline_path else None,
        "baseline_val_metrics": baseline_metrics,
        "train_ids": [sample["ID"] for sample in train_samples],
        "val_ids": [sample["ID"] for sample in val_samples],
        "seed": args.seed,
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Device: {device} | AMP: {amp}")
    print(f"AIHub collision labels: {len(labels)} | Train: {len(train_samples)} | Val: {len(val_samples)}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Run: {run_dir}")
    if baseline_metrics is not None:
        print(
            "MM-AU model on the same val IDs | "
            f"@0.3 {baseline_metrics['acc_03']:.1%} | "
            f"median {baseline_metrics['median_sec']:.3f}s"
        )

    best_acc, best_loss, bad_epochs = -1.0, float("inf"), 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        lr = optimizer.param_groups[0]["lr"]
        for sample_index in rng.permutation(len(train_samples)):
            sample = train_samples[int(sample_index)]
            inputs = sample["x"].unsqueeze(0).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
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

        metrics, predictions = evaluate_collision(
            model, val_samples, loss_fn, device, amp
        )
        scheduler.step(metrics["loss"])
        improved = metrics["acc_03"] > best_acc + 1e-12 or (
            abs(metrics["acc_03"] - best_acc) <= 1e-12
            and metrics["loss"] < best_loss - 1e-6
        )
        if improved:
            best_acc, best_loss, bad_epochs = metrics["acc_03"], metrics["loss"], 0
            torch.save(
                {
                    "model_state_dict": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "model_config": model_config,
                    "input_size": model_config["input_size"],
                    "hidden_size": model_config["hidden_size"],
                    "num_layers": model_config["num_layers"],
                    "dropout": model_config["dropout"],
                    "feature_config": FEATURE_CONFIG,
                    "loss_config": loss_fn.config,
                    "epoch": epoch,
                    "metrics": metrics,
                    "seed": args.seed,
                    "task": "collision_frame_aihub_only",
                },
                best_path,
            )
            predictions.to_csv(
                run_dir / "val_predictions.csv", index=False, encoding="utf-8-sig"
            )
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
            f"Epoch {epoch:02d} | train {row['train_loss']:.4f} | "
            f"val {metrics['loss']:.4f} | @0.3 {metrics['acc_03']:.1%} | "
            f"exact {metrics['exact']:.1%} | median {metrics['median_sec']:.3f}s | {status}"
        )
        if bad_epochs >= args.early_patience:
            print("Early stopping")
            break

    best = torch.load(best_path, map_location="cpu", weights_only=True)
    print(f"Best epoch: {best['epoch']}")
    print(f"Best metrics: {best['metrics']}")
    print(f"Checkpoint: {best_path}")


if __name__ == "__main__":
    main()
