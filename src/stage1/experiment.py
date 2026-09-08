"""Run reproducible Stage 1 experiments on Baidu or manifest-based videos."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from src.common.runtime import default_device, set_seed
from src.stage1.checkpoint import append_metrics, save_checkpoint, save_config
from src.stage1.data import build_baidu_dataloaders, build_direct_dataloaders
from src.stage1.engine import train_one_epoch, validate
from src.stage1.model import build_stage1_model
from src.stage1.optim import build_optimizer, build_scheduler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("baidu", "direct"),
        default="baidu",
        help="Input dataset type; the default preserves the original Baidu command",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Baidu root containing the official train and val folders",
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        help="Direct-video manifest containing train/val split rows",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        help="Optional local root containing source/ and recaptured/",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional local cache for decoded Direct clips",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Output folder for checkpoints, config.json, and metrics.csv",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--val-batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--expected-source-frames", type=int, default=60)
    parser.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
    parser.add_argument("--scheduler", choices=("cosine", "none"), default="cosine")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Stop after this many non-improving epochs; zero disables it",
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use torchvision pretrained weights when no init checkpoint is given",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Optional Stage 1 best.pt used to initialize fine-tuning",
    )
    parser.add_argument(
        "--balanced-sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Balance ORIGINAL/RERECORDED sampling for Direct training",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA mixed precision when a GPU is available",
    )
    return parser.parse_args()


def _resolved_config(args: argparse.Namespace) -> dict:
    config = vars(args).copy()
    for name in (
        "data_dir",
        "model_dir",
        "split_csv",
        "video_root",
        "cache_dir",
        "init_checkpoint",
    ):
        value = getattr(args, name)
        config[name] = str(value.expanduser().resolve()) if value is not None else None
    return config


def _build_dataloaders(args: argparse.Namespace) -> tuple:
    if args.dataset == "baidu":
        if args.data_dir is None:
            raise ValueError("--data-dir is required when --dataset baidu")
        return build_baidu_dataloaders(
            args.data_dir,
            frames=args.frames,
            size=args.size,
            batch_size=args.batch_size,
            val_batch_size=args.val_batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            expected_source_frames=args.expected_source_frames,
        )

    if args.split_csv is None:
        raise ValueError("--split-csv is required when --dataset direct")
    return build_direct_dataloaders(
        args.split_csv,
        video_root=args.video_root,
        cache_dir=args.cache_dir,
        frames=args.frames,
        size=args.size,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        balanced_sampling=args.balanced_sampling,
    )


def _build_model(args: argparse.Namespace) -> tuple[nn.Module, Path | None]:
    if args.init_checkpoint is None:
        return build_stage1_model(pretrained=args.pretrained), None

    checkpoint_path = args.init_checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing initialization checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    for key in ("model", "frames", "size"):
        if key not in checkpoint:
            raise KeyError(f"Checkpoint has no {key!r} key: {checkpoint_path}")
    if int(checkpoint["frames"]) != args.frames:
        raise ValueError(
            f"Checkpoint frames={checkpoint['frames']} but --frames={args.frames}"
        )
    if int(checkpoint["size"]) != args.size:
        raise ValueError(f"Checkpoint size={checkpoint['size']} but --size={args.size}")

    model = build_stage1_model(pretrained=False)
    model.net.load_state_dict(checkpoint["model"], strict=True)
    return model, checkpoint_path


def run(args: argparse.Namespace) -> Path:
    if args.epochs <= 0:
        raise ValueError("epochs must be greater than zero")
    if args.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience cannot be negative")

    model_dir = args.model_dir.expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = model_dir / "metrics.csv"
    if metrics_path.exists():
        raise FileExistsError(
            f"Refusing to mix runs in an existing metrics file: {metrics_path}. "
            "Use a new --model-dir for each experiment."
        )

    config = _resolved_config(args)
    save_config(model_dir, config)
    set_seed(args.seed)
    device = default_device()
    amp_enabled = args.amp and device.type == "cuda"

    train_loader, val_loader = _build_dataloaders(args)

    model, initialization_checkpoint = _build_model(args)
    model = model.to(device)
    optimizer = build_optimizer(
        model,
        name=args.optimizer,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(
        optimizer,
        name=args.scheduler,
        epochs=args.epochs,
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    print(f"device: {device}")
    print(f"dataset: {args.dataset}")
    if args.dataset == "baidu":
        print(f"data: {args.data_dir.expanduser().resolve()}")
    else:
        print(f"split CSV: {args.split_csv.expanduser().resolve()}")
        print(
            "video root: "
            f"{args.video_root.expanduser().resolve() if args.video_root else 'CSV paths'}"
        )
        print(
            "clip cache: "
            f"{args.cache_dir.expanduser().resolve() if args.cache_dir else 'disabled'}"
        )
    print(f"output: {model_dir}")
    print(f"train samples: {len(train_loader.dataset)}")
    print(f"val samples: {len(val_loader.dataset)}")
    if initialization_checkpoint is None:
        print(f"initialization: torchvision pretrained={args.pretrained}")
    else:
        print(f"initialization checkpoint: {initialization_checkpoint}")
    print(f"amp: {amp_enabled}")

    best_score = float("-inf")
    best_path = model_dir / "best.pt"
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        learning_rate = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
            amp=amp_enabled,
        )
        validation = validate(
            model,
            val_loader,
            criterion,
            device,
            amp=amp_enabled,
        )
        if scheduler is not None:
            scheduler.step()

        append_metrics(
            model_dir,
            epoch=epoch,
            learning_rate=learning_rate,
            train_loss=train_loss,
            validation=validation,
        )

        score = validation.metrics.macro_f1
        save_checkpoint(
            model_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            size=args.size,
            frames=args.frames,
            val_macro_f1=score,
            config=config,
        )

        improved = score > best_score
        if improved:
            best_score = score
            epochs_without_improvement = 0
            save_checkpoint(
                best_path,
                model=model,
                # Keep the submission checkpoint small; training state lives in last.pt.
                optimizer=None,
                epoch=epoch,
                size=args.size,
                frames=args.frames,
                val_macro_f1=score,
                config=config,
            )
        else:
            epochs_without_improvement += 1

        metrics = validation.metrics
        print(
            f"epoch {epoch:03d}/{args.epochs:03d} | "
            f"lr {learning_rate:.2e} | "
            f"train_loss {train_loss:.4f} | "
            f"val_loss {validation.loss:.4f} | "
            f"macro_f1 {metrics.macro_f1:.4f} | "
            f"original_f1 {metrics.original_f1:.4f} | "
            f"rerecorded_f1 {metrics.rerecorded_f1:.4f} | "
            f"pred_ratio O/R "
            f"{metrics.predicted_original_ratio:.3f}/"
            f"{metrics.predicted_rerecorded_ratio:.3f}"
            f"{' | best' if improved else ''}"
        )

        patience = args.early_stopping_patience
        if patience and epochs_without_improvement >= patience:
            print(f"early stopping after {patience} non-improving epochs")
            break

    print(f"best Macro-F1: {best_score:.4f}")
    print(f"best checkpoint: {best_path}")
    return best_path


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
