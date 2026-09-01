from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


def video_frames(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"cannot decode: {path}")
    return frames


def crop_tensor(rgb: np.ndarray, size: int = 224) -> torch.Tensor:
    height, width = rgb.shape[:2]
    scale = size / min(height, width)
    resized_height = max(size, round(height * scale))
    resized_width = max(size, round(width * scale))
    rgb = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    top = (resized_height - size) // 2
    left = (resized_width - size) // 2
    crop = rgb[top : top + size, left : left + size].copy()
    return torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0


def load_clip(path: Path, frames: int = 16, center: int | None = None) -> tuple[torch.Tensor, int]:
    video = video_frames(path)
    total = len(video)
    if center is None:
        indices = np.linspace(0, total - 1, frames).round().astype(int)
    else:
        indices = np.clip(center - frames // 2 + np.arange(frames), 0, total - 1)
    clip = torch.stack([crop_tensor(video[int(index)]) for index in indices], dim=1)
    return clip, total
