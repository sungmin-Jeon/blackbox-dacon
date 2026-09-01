import os
import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

SIZE = 224

S1_MEAN = torch.tensor(
    [0.45, 0.45, 0.45]
)[:, None, None, None]

S1_STD = torch.tensor(
    [0.225, 0.225, 0.225]
)[:, None, None, None]

S3_MEAN = torch.tensor(
    [0.45, 0.45, 0.45]
)[:, None, None]

S3_STD = torch.tensor(
    [0.225, 0.225, 0.225]
)[:, None, None]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=int(os.getenv("EPOCHS", "1")),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260825,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = args.data_dir.resolve()
    model_dir = args.model_dir.resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    set_seed(args.seed)

    print("data_dir:", data_dir)
    print("model_dir:", model_dir)
    print("device:", device)
    print("epochs:", args.epochs)

    # 이후 학습 함수에 전달
    # fit_stage1(data_dir, model_dir, device, args.epochs)
    # fit_stage2(data_dir, model_dir, device, args.epochs)
    # fit_stage3(data_dir, model_dir, device, args.epochs)


if __name__ == "__main__":
    main()
"""Command-line entry point for Stage 1 training."""

from src.stage1.train import main


if __name__ == "__main__":
    main()
