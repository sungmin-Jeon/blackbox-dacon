"""Dacon 3-stage baseline inference entry points."""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.video import mvit_v2_s


VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".3gp", ".3gpp", ".wmv"}
ACCEL = ["ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED"]
STEER = ["LEFT", "STRAIGHT", "RIGHT"]
S1_MEAN = torch.tensor([0.45, 0.45, 0.45])[:, None, None, None]
S1_STD = torch.tensor([0.225, 0.225, 0.225])[:, None, None, None]
S3_MEAN = torch.tensor([0.45, 0.45, 0.45])[:, None, None]
S3_STD = torch.tensor([0.225, 0.225, 0.225])[:, None, None]
cv2.setNumThreads(1)


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("This submission requires a CUDA GPU evaluation environment.")
    return torch.device("cuda")


def _video_paths(root: Path):
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXT)


# ---------------------------------------------------------------------------
# Stage 1: MViTv2-S replay classifier
# ---------------------------------------------------------------------------
def _clip_ids(path: Path, frames: int, slot: int, slots: int):
    capture = cv2.VideoCapture(str(path))
    total = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    capture.release()
    center = (slot + 0.5) * total / slots
    start = max(0, min(total - frames, round(center - frames / 2)))
    return np.linspace(start, min(total - 1, start + frames - 1), frames).round().astype(int)


def _decode_stage1_clip(path: Path, size: int, frame_ids):
    capture = cv2.VideoCapture(str(path))
    output = []
    wanted = [int(index) for index in frame_ids]
    capture.set(cv2.CAP_PROP_POS_FRAMES, wanted[0])
    position = wanted[0]

    for index in wanted:
        ok = False
        bgr = None
        while position <= index:
            ok, bgr = capture.read()
            position += 1
            if not ok:
                break
        if not ok or bgr is None:
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        scale = size / min(height, width)
        resized_height = max(size, round(height * scale))
        resized_width = max(size, round(width * scale))
        rgb = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        top = (resized_height - size) // 2
        left = (resized_width - size) // 2
        output.append(rgb[top : top + size, left : left + size])

    capture.release()
    if not output:
        raise ValueError(f"cannot decode video: {path.name}")
    while len(output) < len(wanted):
        output.append(output[-1])

    clip = torch.from_numpy(np.stack(output)).permute(3, 0, 1, 2).float() / 255.0
    return (clip - S1_MEAN) / S1_STD


class _Stage1Clips(Dataset):
    def __init__(self, videos, slots: int, size: int, frames: int) -> None:
        self.videos = videos
        self.slots = slots
        self.size = size
        self.frames = frames

    def __len__(self) -> int:
        return len(self.videos) * self.slots

    def __getitem__(self, index: int):
        video_index, slot = index // self.slots, index % self.slots
        path = self.videos[video_index]
        try:
            clip = _decode_stage1_clip(path, self.size, _clip_ids(path, self.frames, slot, self.slots))
            valid = 1
        except Exception:
            clip = torch.zeros(3, self.frames, self.size, self.size)
            valid = 0
        return clip, video_index, valid


def predict_stage1(data_dir, model_dir):
    device = _device()
    checkpoint = torch.load(Path(model_dir) / "best.pt", map_location="cpu", weights_only=False)
    size = int(checkpoint["size"])
    frames = int(checkpoint["frames"])
    model = mvit_v2_s(weights=None)
    model.head[1] = nn.Linear(model.head[1].in_features, 2)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    videos = _video_paths(Path(data_dir) / "videos")
    slots = 3
    dataset = _Stage1Clips(videos, slots, size, frames)
    loader = DataLoader(dataset, batch_size=4, num_workers=4, pin_memory=True)
    scores = [[] for _ in videos]

    with torch.inference_mode():
        for clips, video_indices, valid in loader:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                probabilities = torch.softmax(model(clips.to(device, non_blocking=True)), 1)[:, 1]
            for index, value, ok in zip(video_indices.tolist(), probabilities.float().cpu().tolist(), valid.tolist()):
                if ok:
                    scores[index].append(float(value))

    rows = []
    for path, values in zip(videos, scores):
        probability = float(np.mean(values)) if values else 1.0
        rows.append({"ID": path.stem, "answer": "RERECORDED" if probability >= 0.5 else "ORIGINAL"})

    del model
    torch.cuda.empty_cache()
    return pd.DataFrame(rows, columns=["ID", "answer"])


# ---------------------------------------------------------------------------
# Stage 2: ResNet18 + BiGRU
# ---------------------------------------------------------------------------
class _Stage2Frames(Dataset):
    def __init__(self, paths, transform) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


class _Stage2Temporal(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.r = nn.GRU(512, 192, 2, batch_first=True, bidirectional=True, dropout=0.15)
        self.tc = nn.Linear(384, 1)
        self.te = nn.Linear(384, 1)
        self.scene = nn.Sequential(nn.Linear(768, 192), nn.ReLU(), nn.Dropout(0.2), nn.Linear(192, 4))

    def forward(self, inputs: torch.Tensor):
        hidden, _ = self.r(inputs)
        collision_logits = self.tc(hidden).squeeze(-1)
        entry_logits = self.te(hidden).squeeze(-1)
        collision_index = collision_logits.argmax(1)
        entry_index = entry_logits.argmax(1)
        batch = torch.arange(len(hidden), device=hidden.device)
        scene_input = torch.cat([hidden[batch, collision_index], hidden[batch, entry_index]], 1)
        return collision_index, entry_index, self.scene(scene_input)


def _frame_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else 0


def predict_stage2(data_dir, model_dir):
    device = _device()
    model_dir = Path(model_dir)
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    backbone = resnet18(weights=None)
    backbone.load_state_dict(
        torch.load(model_dir / "resnet18-f37072fd.pth", map_location="cpu", weights_only=True)
    )
    backbone.fc = nn.Identity()
    backbone.to(device).eval()

    temporal = _Stage2Temporal()
    temporal.load_state_dict(torch.load(model_dir / "best.pt", map_location="cpu", weights_only=False)["model"])
    temporal.to(device).eval()

    folders = sorted(path for path in (Path(data_dir) / "images").iterdir() if path.is_dir())
    rows = []
    with torch.inference_mode():
        for folder in folders:
            paths = sorted(
                (path for path in folder.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}),
                key=_frame_number,
            )
            if not paths:
                continue

            loader = DataLoader(_Stage2Frames(paths, transform), batch_size=256, num_workers=6, pin_memory=True)
            features = []
            for images in loader:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    features.append(backbone(images.to(device, non_blocking=True)).float().cpu())

            sequence = torch.cat(features)[None].to(device)
            collision_index, entry_index, scene = temporal(sequence)
            frame_numbers = [_frame_number(path) for path in paths]
            rows.append(
                {
                    "ID": folder.name,
                    "collision_frame": frame_numbers[int(collision_index)],
                    "entry_frame": frame_numbers[int(entry_index)],
                    "evasion_space": int(scene[:, :2].argmax(1)),
                    "entry_side": "RIGHT" if int(scene[:, 2:].argmax(1)) else "LEFT",
                }
            )

    del backbone, temporal
    torch.cuda.empty_cache()
    return pd.DataFrame(rows, columns=["ID", "collision_frame", "entry_frame", "evasion_space", "entry_side"])


# ---------------------------------------------------------------------------
# Stage 3: MViTv2-S multi-head classifier
# ---------------------------------------------------------------------------
class _Stage3MViT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = mvit_v2_s(weights=None)
        dimension = self.backbone.head[1].in_features
        self.backbone.head = nn.Identity()
        self.accel = nn.Linear(dimension, 4)
        self.steer = nn.Linear(dimension, 3)

    def forward(self, inputs: torch.Tensor):
        features = self.backbone(inputs)
        return self.accel(features), self.steer(features)


def _stage3_frames(path: Path):
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        width, height = image.size
        scale = 256 / min(width, height)
        image = image.resize((round(width * scale), round(height * scale)))
        width, height = image.size
        left, top = (width - 224) // 2, (height - 224) // 2
        image = image.crop((left, top, left + 224, top + 224))
        frames.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).to(torch.uint8))
    capture.release()

    if not frames:
        raise ValueError(f"cannot decode video: {path.name}")
    return torch.stack(frames)


def predict_stage3(data_dir, model_dir):
    device = _device()
    checkpoint = torch.load(Path(model_dir) / "best.pt", map_location="cpu", weights_only=False)
    model = _Stage3MViT()
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    videos = _video_paths(Path(data_dir) / "videos")
    rows = []

    with torch.inference_mode():
        for path in videos:
            frames = _stage3_frames(path)
            count = len(frames)
            centers = np.arange(count)
            accel_predictions = []
            steer_predictions = []

            for start in range(0, count, 8):
                center = centers[start : start + 8]
                indices = np.clip(center[:, None] - 8 + np.arange(16)[None, :], 0, count - 1)
                clips = frames[torch.from_numpy(indices)].permute(0, 2, 1, 3, 4).float() / 255.0
                clips = (clips - S3_MEAN[None, :, None, :, :]) / S3_STD[None, :, None, :, :]
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    accel_logits, steer_logits = model(clips.to(device, non_blocking=True))
                accel_predictions.extend(accel_logits.argmax(1).cpu().tolist())
                steer_predictions.extend(steer_logits.argmax(1).cpu().tolist())

            for sample_index, (accel, steer) in enumerate(zip(accel_predictions, steer_predictions)):
                rows.append(
                    {
                        "ID": path.stem,
                        "sample_index": sample_index,
                        "accel_label": ACCEL[accel],
                        "steer_label": STEER[steer],
                    }
                )

    del model
    torch.cuda.empty_cache()
    return pd.DataFrame(rows, columns=["ID", "sample_index", "accel_label", "steer_label"])
