"""Train any combination of the three baseline stages."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.common.runtime import default_device, set_seed
from src.stage1.train import train as train_stage1
from src.stage2.train import train as train_stage2
from src.stage3.train import train as train_stage3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Root containing stage1, stage2 and stage3")
    parser.add_argument("--model-dir", type=Path, required=True, help="Root in which Stage checkpoints are saved")
    parser.add_argument("--epochs", type=int, default=int(os.getenv("EPOCHS", "1")))
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument(
        "--stages",
        type=int,
        nargs="+",
        choices=(1, 2, 3),
        default=(1, 2, 3),
        help="Stages to train; the default trains all three",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_dir.expanduser().resolve()
    model_root = args.model_dir.expanduser().resolve()
    stages = set(args.stages)
    set_seed(args.seed)
    device = default_device()

    print(f"device: {device}")
    print(f"data root: {data_root}")
    print(f"model root: {model_root}")

    if 1 in stages:
        checkpoint = train_stage1(
            data_root / "stage1",
            model_root / "stage1",
            device,
            args.epochs,
            args.seed,
        )
        print(f"Stage 1 complete: {checkpoint}")

    if 2 in stages:
        checkpoint, backbone = train_stage2(
            data_root / "stage2",
            model_root / "stage2",
            device,
            args.epochs,
        )
        print(f"Stage 2 complete: {checkpoint}")
        print(f"Stage 2 backbone: {backbone}")

    if 3 in stages:
        checkpoint = train_stage3(
            data_root / "stage3",
            model_root / "stage3",
            device,
            args.epochs,
        )
        print(f"Stage 3 complete: {checkpoint}")


if __name__ == "__main__":
    main()
