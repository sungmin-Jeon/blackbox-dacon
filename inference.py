"""Dacon 3-stage baseline inference entry points."""

from __future__ import annotations

import re
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
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
    """Legacy clip selection retained for old callers and checkpoints."""
    capture = cv2.VideoCapture(str(path))
    total = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    capture.release()
    if slots == 1:
        return np.linspace(0, total - 1, frames).round().astype(int)
    center = (slot + 0.5) * total / slots
    start = max(0, min(total - frames, round(center - frames / 2)))
    return np.linspace(start, min(total - 1, start + frames - 1), frames).round().astype(int)


TEMPORAL_VERSION = 2
SUPPORTED_TEMPORAL_VERSIONS = (1, TEMPORAL_VERSION)
TEMPORAL_TRAIN_MODES = ("uniform", "multi-burst", "mixed")
TEMPORAL_EVAL_MODES = ("uniform", "multi-burst", "both")


def _temporal_config(mode="uniform", eval_mode="auto", bursts=4,
                     target_fps=15.0, version=TEMPORAL_VERSION):
    """Versioned Stage 1 temporal sampling configuration.

    ``mode`` controls Direct training. ``eval_mode`` controls validation and
    submission; ``both`` averages a global-uniform and a multi-burst view.
    Old checkpoints resolve to one uniform view.
    """
    if mode not in TEMPORAL_TRAIN_MODES:
        raise ValueError(f"Unknown temporal training mode: {mode}")
    if eval_mode in (None, "auto"):
        eval_mode = {
            "uniform": "uniform",
            "multi-burst": "multi-burst",
            "mixed": "both",
        }[mode]
    if eval_mode not in TEMPORAL_EVAL_MODES:
        raise ValueError(f"Unknown temporal evaluation mode: {eval_mode}")
    if bursts < 1:
        raise ValueError("temporal bursts must be at least one")
    if not np.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("temporal target FPS must be greater than zero")
    if version not in SUPPORTED_TEMPORAL_VERSIONS:
        raise ValueError(f"Unsupported temporal preprocessing version: {version}")
    return dict(
        mode=mode,
        eval_mode=eval_mode,
        bursts=int(bursts),
        target_fps=float(target_fps),
        version=int(version),
    )


def _stage1_temporal_config(checkpoint):
    """Restore temporal settings while keeping old checkpoints unchanged."""
    saved = checkpoint.get("config", {}).get("temporal")
    if saved is None:
        return _temporal_config(version=1)
    return _temporal_config(**saved)


def _temporal_eval_views(temporal):
    options = _temporal_config(**(temporal or {}))
    if options["eval_mode"] == "both":
        return ("uniform", "multi-burst")
    return (options["eval_mode"],)


def _validate_temporal_frames(frames, temporal):
    """Validate frame-count-dependent temporal settings before decoding."""
    options = _temporal_config(**(temporal or {}))
    frames = int(frames)
    if frames < 1:
        raise ValueError("frames must be at least one")
    uses_multi_burst = (
        options["mode"] in {"multi-burst", "mixed"}
        or options["eval_mode"] in {"multi-burst", "both"}
    )
    if uses_multi_burst and frames % options["bursts"]:
        raise ValueError(
            f"frames={frames} must be divisible by temporal bursts={options['bursts']}"
        )
    return options


def _temporal_cache_tag(temporal):
    temporal = _temporal_config(**temporal)
    digest = hashlib.sha256(json.dumps(temporal, sort_keys=True).encode()).hexdigest()[:16]
    return (
        f"temporal_v{temporal['version']}_{temporal['mode']}_"
        f"{temporal['eval_mode']}_{digest}"
    )


def _temporal_frame_ids(total, frames, *, mode="uniform", bursts=4,
                        training=False, fps=None, target_fps=15.0,
                        version=TEMPORAL_VERSION):
    """Select one chronological Stage 1 view from a known frame count.

    Multi-burst splits the whole video into ``bursts`` segments. Version 2
    always centers each burst and spaces its frames at a target-rate-normalized
    stride. For example, target 15 FPS uses raw strides 1/2/4 for videos stored
    at 15/30/60 FPS. Version 1 retains the original consecutive-frame training
    jitter for existing checkpoints. Empty/short segments repeat valid indices
    so the output length remains exactly ``frames``.
    """
    total = max(1, int(total))
    frames = int(frames)
    bursts = int(bursts)
    if frames < 1:
        raise ValueError("frames must be at least one")
    if bursts < 1:
        raise ValueError("temporal bursts must be at least one")
    if not np.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("temporal target FPS must be greater than zero")
    if version not in SUPPORTED_TEMPORAL_VERSIONS:
        raise ValueError(f"Unsupported temporal preprocessing version: {version}")

    if mode == "mixed":
        if not training:
            raise ValueError("mixed temporal sampling is training-only")
        mode = "uniform" if int(np.random.randint(2)) == 0 else "multi-burst"
    if mode == "uniform":
        return np.linspace(0, total - 1, frames).round().astype(int)
    if mode != "multi-burst":
        raise ValueError(f"Unknown temporal view: {mode}")
    if frames % bursts:
        raise ValueError(
            f"frames={frames} must be divisible by temporal bursts={bursts}"
        )

    frames_per_burst = frames // bursts
    source_fps = float(fps) if fps is not None else float(target_fps)
    if not np.isfinite(source_fps) or source_fps <= 0:
        source_fps = float(target_fps)
    frame_stride = (
        1
        if version == 1
        else max(1, int(round(source_fps / float(target_fps))))
    )
    boundaries = np.linspace(0, total, bursts + 1).astype(int)
    selected = []
    for burst_index in range(bursts):
        first = int(boundaries[burst_index])
        stop = int(boundaries[burst_index + 1])
        segment_length = stop - first

        required_length = 1 + (frames_per_burst - 1) * frame_stride
        if segment_length >= required_length:
            latest_start = stop - required_length
            if version == 1 and training:
                start = int(np.random.randint(first, latest_start + 1))
            else:
                start = first + (segment_length - required_length) // 2
            burst_ids = start + np.arange(frames_per_burst, dtype=int) * frame_stride
        elif segment_length > 0:
            burst_ids = np.linspace(
                first, stop - 1, frames_per_burst
            ).round().astype(int)
        else:
            midpoint = round((burst_index + 0.5) * total / bursts - 0.5)
            burst_ids = np.full(
                frames_per_burst, np.clip(midpoint, 0, total - 1), dtype=int
            )
        selected.append(burst_ids)

    return np.concatenate(selected).astype(int)


def _temporal_clip_ids(path: Path, frames: int, *, temporal=None,
                       training=False, view=None):
    """Read a video's frame count/FPS and select one configured temporal view."""
    options = _temporal_config(**(temporal or {}))
    mode = options["mode"] if view is None else view
    capture = cv2.VideoCapture(str(path))
    total = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    return _temporal_frame_ids(
        total,
        frames,
        mode=mode,
        bursts=options["bursts"],
        training=training,
        fps=fps,
        target_fps=options["target_fps"],
        version=options["version"],
    )


SPATIAL_VERSION = 1
SPATIAL_MODES = ("center", "native-center", "random", "fft")


def _spatial_config(mode="center", crop_size=224, grid_size=5, seed=42,
                    fft_min_freq=0.25, version=SPATIAL_VERSION):
    """Versioned preprocessing, shared by training, evaluation and submission."""
    if mode not in SPATIAL_MODES:
        raise ValueError(f"Unknown spatial mode: {mode}")
    if crop_size < 2 or grid_size < 1:
        raise ValueError("crop_size must be >= 2 and grid_size must be >= 1")
    if not 0 < fft_min_freq < 0.5:
        raise ValueError("fft_min_freq must be between 0 and 0.5 cycles/pixel")
    if version != SPATIAL_VERSION:
        raise ValueError(f"Unsupported spatial preprocessing version: {version}")
    return dict(mode=mode, crop_size=int(crop_size), grid_size=int(grid_size),
                seed=int(seed), fft_min_freq=float(fft_min_freq), version=version)


def _stage1_spatial_config(checkpoint):
    """Old checkpoints retain the original resize + center-crop behavior."""
    return _spatial_config(**checkpoint.get("config", {}).get("spatial", {}))


def _spatial_cache_tag(spatial):
    spatial = _spatial_config(**spatial)
    digest = hashlib.sha256(json.dumps(spatial, sort_keys=True).encode()).hexdigest()[:16]
    return f"spatial_v{SPATIAL_VERSION}_{spatial['mode']}_{digest}"


def _crop_candidates(height, width, crop_size, grid_size):
    if min(height, width) < crop_size:
        raise ValueError(f"Native crop {crop_size} exceeds frame {width}x{height}")
    ys = np.unique(np.linspace(0, height - crop_size, grid_size).round().astype(int))
    xs = np.unique(np.linspace(0, width - crop_size, grid_size).round().astype(int))
    return [(int(y), int(x)) for y in ys for x in xs]


def _fft_patch_score(rgb, min_freq=0.25):
    """Mean high-frequency power of mean-centered, Hann-windowed luminance.

    Frequency units are cycles/pixel, using fftfreq (NOT array-index order).
    This radial-band score is a video-experiment adaptation, not a reproduction
    of the DGOAS paper's single-bin patch score. It is not a replay probability.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    height, width = gray.shape
    window = np.outer(np.hanning(height), np.hanning(width))
    spectrum = np.fft.fft2((gray - gray.mean()) * window, norm="ortho")
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    mask = np.hypot(fy, fx) >= min_freq
    return float(np.mean(np.abs(spectrum[mask]) ** 2))


def _read_stage1_frames(path, wanted):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"cannot open video: {path.name}")
    try:
        for index in wanted:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, bgr = capture.read()
            if ok and bgr is not None:
                yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def _decode_stage1_clip(path: Path, size: int, frame_ids, *, spatial=None,
                        training=False):
    """One spatially consistent clip, always [3, len(frame_ids), size, size].

    FFT uses a streaming scoring pass then a crop pass, avoiding retention of
    all full-resolution frames in RAM. Random validation is filename/seed based
    so moving the same files between Colab and Drive does not change crops.
    """
    path = Path(path)
    wanted = [int(index) for index in frame_ids]
    if not wanted or size <= 0:
        raise ValueError("frame_ids must be nonempty and size must be positive")
    options = _spatial_config(**(spatial or {}))
    mode = options["mode"]
    crop_size = options["crop_size"]
    coordinates = None
    frame_shape = None

    if mode == "fft":
        scores = None
        candidates = None
        for rgb in _read_stage1_frames(path, wanted):
            if frame_shape is None:
                frame_shape = rgb.shape[:2]
                candidates = _crop_candidates(*frame_shape, crop_size, options["grid_size"])
                scores = np.zeros(len(candidates), dtype=np.float64)
            elif rgb.shape[:2] != frame_shape:
                raise ValueError(f"Frame dimensions change within {path.name}")
            for i, (top, left) in enumerate(candidates):
                scores[i] += _fft_patch_score(
                    rgb[top:top + crop_size, left:left + crop_size],
                    options["fft_min_freq"],
                )
        if scores is None:
            raise ValueError(f"cannot decode video: {path.name}")
        # Every candidate sees the same frames: sum and mean give the same rank.
        # np.argmax deterministically selects the first candidate on a tie.
        coordinates = candidates[int(np.argmax(scores))]

    output = []
    for rgb in _read_stage1_frames(path, wanted):
        if mode == "center":
            # Preserve the historical preprocessing exactly for old experiments.
            height, width = rgb.shape[:2]
            scale = size / min(height, width)
            resized_height = max(size, round(height * scale))
            resized_width = max(size, round(width * scale))
            rgb = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
            top = (resized_height - size) // 2
            left = (resized_width - size) // 2
            output.append(rgb[top : top + size, left : left + size])
            continue

        if frame_shape is None:
            frame_shape = rgb.shape[:2]
        elif rgb.shape[:2] != frame_shape:
            raise ValueError(f"Frame dimensions change within {path.name}")
        if coordinates is None:
            height, width = frame_shape
            candidates = _crop_candidates(height, width, crop_size, options["grid_size"])
            if mode == "native-center":
                coordinates = ((height - crop_size) // 2, (width - crop_size) // 2)
            elif training:
                coordinates = candidates[int(np.random.randint(len(candidates)))]
            else:
                key = f"{options['seed']}:{path.name}".encode()
                seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
                rng = np.random.default_rng(seed)
                coordinates = candidates[int(rng.integers(len(candidates)))]
        top, left = coordinates
        patch = rgb[top:top + crop_size, left:left + crop_size]
        if crop_size != size:
            patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
        output.append(patch)

    if not output:
        raise ValueError(f"cannot decode video: {path.name}")
    while len(output) < len(wanted):
        output.append(output[-1])

    clip = torch.from_numpy(np.stack(output)).permute(3, 0, 1, 2).float() / 255.0
    return (clip - S1_MEAN) / S1_STD


class _Stage1Clips(Dataset):
    def __init__(self, videos, slots: int, size: int, frames: int, spatial=None,
                 temporal=None) -> None:
        self.videos = videos
        self.slots = slots
        self.size = size
        self.frames = frames
        self.spatial = _spatial_config(**(spatial or {}))
        self.temporal = (
            _temporal_config(**temporal) if temporal is not None else None
        )
        self.temporal_views = (
            _temporal_eval_views(self.temporal) if self.temporal is not None else None
        )
        if self.temporal is not None:
            _validate_temporal_frames(self.frames, self.temporal)
        if self.temporal_views is not None and self.slots != len(self.temporal_views):
            raise ValueError(
                f"slots={slots} does not match temporal views={self.temporal_views}"
            )

    def __len__(self) -> int:
        return len(self.videos) * self.slots

    def __getitem__(self, index: int):
        video_index, slot = index // self.slots, index % self.slots
        path = self.videos[video_index]
        try:
            if self.temporal_views is None:
                frame_ids = _clip_ids(path, self.frames, slot, self.slots)
            else:
                frame_ids = _temporal_clip_ids(
                    path, self.frames, temporal=self.temporal,
                    view=self.temporal_views[slot],
                )
            clip = _decode_stage1_clip(
                path, self.size, frame_ids, spatial=self.spatial,
            )
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
    temporal = _stage1_temporal_config(checkpoint)
    slots = len(_temporal_eval_views(temporal))
    dataset = _Stage1Clips(
        videos, slots, size, frames, _stage1_spatial_config(checkpoint), temporal,
    )
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
# Stage 2: collision BiGRU + direct spatial-attention BiGRU
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


class _Stage2RawFrames(Dataset):
    def __init__(self, paths) -> None:
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            rgb = np.asarray(image.convert("RGB")).copy()
        return torch.from_numpy(rgb).permute(2, 0, 1)


class _Stage2CollisionBiGRU(nn.Module):
    def __init__(
        self,
        input_size: int = 512,
        hidden_size: int = 192,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.collision_head = nn.Linear(hidden_size * 2, 1)

    def forward(self, inputs: torch.Tensor):
        hidden, _ = self.gru(inputs)
        return self.collision_head(self.dropout(hidden)).squeeze(-1)


class _Stage2BaselineCollisionBiGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.r = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.15 if num_layers > 1 else 0.0,
        )
        self.tc = nn.Linear(hidden_size * 2, 1)

    def forward(self, inputs: torch.Tensor):
        hidden, _ = self.r(inputs)
        return self.tc(hidden).squeeze(-1)


def _stage2_checkpoint_state(checkpoint):
    for key in ("model_state_dict", "model", "state_dict"):
        if key in checkpoint:
            state = checkpoint[key]
            break
    else:
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            state = checkpoint
        else:
            raise ValueError("Stage 2 collision checkpoint has no model state dictionary")
    return {
        (name.removeprefix("module.")): value
        for name, value in state.items()
    }


def _stage2_gru_dimensions(state, prefix):
    input_key = f"{prefix}.weight_ih_l0"
    hidden_key = f"{prefix}.weight_hh_l0"
    if input_key not in state or hidden_key not in state:
        raise ValueError(f"Missing GRU weights for prefix {prefix}")
    input_size = int(state[input_key].shape[1])
    hidden_size = int(state[hidden_key].shape[1])
    layer_pattern = re.compile(rf"^{re.escape(prefix)}\.weight_ih_l(\d+)$")
    layers = [int(match.group(1)) for key in state if (match := layer_pattern.match(key))]
    if not layers:
        raise ValueError(f"Cannot infer GRU layers for prefix {prefix}")
    return input_size, hidden_size, max(layers) + 1


def _stage2_collision_model_from_checkpoint(checkpoint):
    state = _stage2_checkpoint_state(checkpoint)
    if "gru.weight_ih_l0" in state and "collision_head.weight" in state:
        input_size, hidden_size, num_layers = _stage2_gru_dimensions(state, "gru")
        model = _Stage2CollisionBiGRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=float(checkpoint.get("dropout", 0.3)),
        )
        model.load_state_dict(
            {
                key: value
                for key, value in state.items()
                if key.startswith("gru.") or key.startswith("collision_head.")
            }
        )
        return model, "collision_v0"
    if "r.weight_ih_l0" in state and "tc.weight" in state:
        input_size, hidden_size, num_layers = _stage2_gru_dimensions(state, "r")
        model = _Stage2BaselineCollisionBiGRU(input_size, hidden_size, num_layers)
        model.load_state_dict(
            {
                key: value
                for key, value in state.items()
                if key.startswith("r.") or key.startswith("tc.")
            }
        )
        return model, "baseline"
    raise ValueError(
        "Unsupported Stage 2 collision checkpoint keys: "
        f"{list(state)[:10]}"
    )


class _Stage2SpatialAttentionPool(nn.Module):
    def __init__(
        self,
        input_channels: int = 256,
        projection_size: int = 128,
        output_size: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(input_channels, projection_size, kernel_size=1),
            nn.GELU(),
        )
        attention_channels = max(16, projection_size // 2)
        self.attention = nn.Sequential(
            nn.Conv2d(projection_size, attention_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(attention_channels, 1, kernel_size=1),
        )
        self.fuse = nn.Sequential(
            nn.Linear(projection_size * 2, output_size),
            nn.LayerNorm(output_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, maps: torch.Tensor):
        batch, steps, channels, height, width = maps.shape
        flat = maps.reshape(batch * steps, channels, height, width)
        projected = self.project(flat)
        weights = self.attention(projected).flatten(2).softmax(dim=-1)
        local = torch.bmm(projected.flatten(2), weights.transpose(1, 2)).squeeze(-1)
        global_scene = projected.mean(dim=(-2, -1))
        vectors = self.fuse(torch.cat([local, global_scene], dim=-1))
        return vectors.reshape(batch, steps, -1)


class _Stage2DirectSpatial(nn.Module):
    def __init__(
        self,
        input_channels: int = 256,
        projection_size: int = 128,
        temporal_input_size: int = 256,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
        collision_window: int = 2,
        entry_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.collision_window = collision_window
        self.entry_temperature = entry_temperature
        self.spatial = _Stage2SpatialAttentionPool(
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
        dimension = hidden_size * 2
        self.dropout = nn.Dropout(dropout)
        self.entry_head = nn.Linear(dimension, 1)
        self.side_head = nn.Sequential(
            nn.Linear(dimension, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )
        self.evasion_head = nn.Sequential(
            nn.Linear(dimension * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )

    def _collision_context(self, hidden: torch.Tensor, collision_indices: torch.Tensor):
        batch, steps, _ = hidden.shape
        indices = collision_indices.to(device=hidden.device, dtype=torch.long).reshape(-1)
        if len(indices) != batch or (indices < -1).any() or (indices >= steps).any():
            raise ValueError("collision_indices must contain an index or -1 per video")
        contexts = []
        for batch_index, center in enumerate(indices.tolist()):
            if center == -1:
                contexts.append(hidden[batch_index].mean(dim=0))
            else:
                start = max(0, center - self.collision_window)
                end = min(steps, center + self.collision_window + 1)
                contexts.append(hidden[batch_index, start:end].mean(dim=0))
        return torch.stack(contexts)

    def forward(self, maps: torch.Tensor, collision_indices: torch.Tensor):
        vectors = self.spatial(maps)
        hidden, _ = self.temporal(vectors)
        hidden = self.dropout(hidden)
        entry_logits = self.entry_head(hidden).squeeze(-1)
        entry_weights = (entry_logits / self.entry_temperature).softmax(dim=1)
        entry_context = torch.bmm(entry_weights.unsqueeze(1), hidden).squeeze(1)
        side_logits = self.side_head(entry_context)
        collision_context = self._collision_context(hidden, collision_indices)
        evasion_logits = self.evasion_head(
            torch.cat([collision_context, hidden.mean(dim=1)], dim=-1)
        )
        return entry_logits, side_logits, evasion_logits


class _Stage2FeatureTemporal(nn.Module):
    """Submission copy of the cached spatial-feature temporal model."""

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
        delta_mode: str = "none",
        spatial_coordinates: bool = False,
    ) -> None:
        super().__init__()
        if feature_kind != "spatial":
            raise ValueError("Final Stage 2 feature models require spatial features")
        if delta_mode not in {"none", "concat"}:
            raise ValueError(f"Unsupported final Stage 2 delta mode: {delta_mode}")
        if spatial_coordinates:
            raise ValueError("Final Stage 2 model does not use coordinate summaries")
        self.delta_mode = delta_mode
        projected_channels = input_channels * (2 if delta_mode == "concat" else 1)
        self.project = _Stage2SpatialAttentionPool(
            input_channels=projected_channels,
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

    @staticmethod
    def temporal_delta(features: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [torch.zeros_like(features[:, :1]), features[:, 1:] - features[:, :-1]],
            dim=1,
        )

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        if self.delta_mode == "concat":
            features = torch.cat([features, self.temporal_delta(features)], dim=2)
        vectors = self.project(features)
        hidden, _ = self.temporal(vectors)
        return hidden

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(features)
        return self.collision_head(self.dropout(hidden)).squeeze(-1)


class _Stage2EntrySideTemporal(_Stage2FeatureTemporal):
    def __init__(
        self,
        *,
        entry_temperature: float = 1.0,
        side_context: str = "entry",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if side_context not in {"entry", "independent"}:
            raise ValueError(f"Unsupported Side context: {side_context}")
        self.entry_temperature = entry_temperature
        self.side_context = side_context
        hidden_size = int(kwargs.get("hidden_size", 64))
        dropout = float(kwargs.get("dropout", 0.3))
        self.side_attention = (
            nn.Linear(hidden_size * 2, 1)
            if side_context == "independent"
            else None
        )
        self.side_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, features: torch.Tensor):
        hidden = self.encode(features)
        dropped = self.dropout(hidden)
        entry_logits = self.collision_head(dropped).squeeze(-1)
        if self.side_context == "entry":
            side_weights = (entry_logits / self.entry_temperature).softmax(dim=1)
        else:
            side_weights = self.side_attention(dropped).squeeze(-1).softmax(dim=1)
        side_context = torch.bmm(side_weights.unsqueeze(1), dropped).squeeze(1)
        return entry_logits, self.side_head(side_context)


def _stage2_final_models(model_dir: Path, device: torch.device):
    paths = {
        name: model_dir / f"{name}.pt"
        for name in ("collision", "entry", "side", "evasion")
    }
    if not all(path.is_file() for path in paths.values()):
        return None
    checkpoints = {
        name: torch.load(path, map_location="cpu", weights_only=False)
        for name, path in paths.items()
    }
    collision = _Stage2FeatureTemporal(**checkpoints["collision"]["model_config"])
    collision.load_state_dict(checkpoints["collision"]["model_state_dict"])
    entry = _Stage2EntrySideTemporal(**checkpoints["entry"]["model_config"])
    entry.load_state_dict(checkpoints["entry"]["model_state_dict"])
    side = _Stage2EntrySideTemporal(**checkpoints["side"]["model_config"])
    side.load_state_dict(checkpoints["side"]["model_state_dict"])
    evasion = _Stage2DirectSpatial(**checkpoints["evasion"]["model_config"])
    evasion.load_state_dict(checkpoints["evasion"]["model_state_dict"])
    models = {
        "collision": collision.to(device).eval(),
        "entry": entry.to(device).eval(),
        "side": side.to(device).eval(),
        "evasion": evasion.to(device).eval(),
    }
    return models, checkpoints


def _stage2_final_feature_config(checkpoints):
    configs = [checkpoint.get("feature_config", {}) for checkpoint in checkpoints.values()]
    layer = configs[0].get("layer", "layer3")
    short_edge = int(configs[0].get("short_edge", 256))
    mean = tuple(configs[0].get("mean", (0.485, 0.456, 0.406)))
    std = tuple(configs[0].get("std", (0.229, 0.224, 0.225)))
    for config in configs:
        current = (
            config.get("layer", "layer3"),
            int(config.get("short_edge", 256)),
            tuple(config.get("mean", mean)),
            tuple(config.get("std", std)),
            config.get("crop"),
        )
        if current != (layer, short_edge, mean, std, None):
            raise ValueError("Final Stage 2 checkpoints use different feature settings")
    if layer not in {"layer3", "layer4"} or short_edge < 32:
        raise ValueError("Unsupported final Stage 2 feature settings")
    return layer, short_edge, mean, std


def _predict_stage2_final(data_dir, backbone, models, checkpoints, device):
    layer, short_edge, mean_values, std_values = _stage2_final_feature_config(
        checkpoints
    )
    mean = torch.tensor(mean_values, device=device).view(1, 3, 1, 1)
    std = torch.tensor(std_values, device=device).view(1, 3, 1, 1)
    expected_channels = 256 if layer == "layer3" else 512
    for name in ("collision", "entry", "side"):
        actual = int(checkpoints[name]["model_config"]["input_channels"])
        if actual != expected_channels:
            raise ValueError(f"Stage 2 {name} channels do not match {layer}")
    evasion_channels = int(checkpoints["evasion"]["model_config"]["input_channels"])
    if evasion_channels != expected_channels:
        raise ValueError("Stage 2 evasion channels do not match feature layer")

    folders = sorted(
        path for path in (Path(data_dir) / "images").iterdir() if path.is_dir()
    )
    rows = []
    with torch.inference_mode():
        for folder in folders:
            paths = sorted(
                (
                    path
                    for path in folder.iterdir()
                    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
                ),
                key=_frame_number,
            )
            if not paths:
                continue
            loader = DataLoader(
                _Stage2RawFrames(paths),
                batch_size=64,
                num_workers=6,
                pin_memory=True,
            )
            spatial_maps = []
            for uint8_images in loader:
                images = uint8_images.to(
                    device=device, dtype=torch.float32, non_blocking=True
                ).div_(255)
                images = _stage2_resize_short_edge(images, short_edge)
                images = (images - mean) / std
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    maps = _stage2_spatial_features(backbone, images, layer)
                spatial_maps.append(maps.half().cpu())
                del images, maps

            sequence = torch.cat(spatial_maps)[None].to(
                device=device, dtype=torch.float32
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                collision_logits = models["collision"](sequence)
                entry_logits, _ = models["entry"](sequence)
                _, side_logits = models["side"](sequence)
                collision_index = int(collision_logits.argmax(1).item())
                _, _, evasion_logits = models["evasion"](
                    sequence,
                    torch.tensor([collision_index], device=device),
                )
            entry_index = int(entry_logits.argmax(1).item())
            entry_side = "RIGHT" if int(side_logits.argmax(1).item()) == 1 else "LEFT"
            evasion_space = int(evasion_logits.argmax(1).item())
            frame_numbers = [_frame_number(path) for path in paths]
            rows.append(
                {
                    "ID": folder.name,
                    "collision_frame": frame_numbers[collision_index],
                    "entry_frame": frame_numbers[entry_index],
                    "evasion_space": evasion_space,
                    "entry_side": entry_side,
                }
            )
            del sequence, spatial_maps, collision_logits, entry_logits
            del side_logits, evasion_logits
    return pd.DataFrame(
        rows,
        columns=["ID", "collision_frame", "entry_frame", "evasion_space", "entry_side"],
    )


def _frame_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else 0


def _stage2_resize_short_edge(images: torch.Tensor, short_edge: int):
    height, width = images.shape[-2:]
    scale = short_edge / min(height, width)
    return F.interpolate(
        images,
        size=(max(1, round(height * scale)), max(1, round(width * scale))),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )


def _stage2_spatial_features(backbone: nn.Module, images: torch.Tensor, layer: str):
    hidden = backbone.conv1(images)
    hidden = backbone.bn1(hidden)
    hidden = backbone.relu(hidden)
    hidden = backbone.maxpool(hidden)
    hidden = backbone.layer1(hidden)
    hidden = backbone.layer2(hidden)
    hidden = backbone.layer3(hidden)
    if layer == "layer4":
        hidden = backbone.layer4(hidden)
    return hidden


def predict_stage2(data_dir, model_dir):
    device = _device()
    model_dir = Path(model_dir)

    # Match the center-crop ImageNet preprocessing used to build the V0 cache.
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    backbone = resnet18(weights=None)
    backbone.load_state_dict(
        torch.load(model_dir / "resnet18-f37072fd.pth", map_location="cpu", weights_only=True)
    )
    backbone.fc = nn.Identity()
    backbone.to(device).eval()

    final = _stage2_final_models(model_dir, device)
    if final is not None:
        models, checkpoints = final
        result = _predict_stage2_final(
            data_dir, backbone, models, checkpoints, device
        )
        del backbone, models
        torch.cuda.empty_cache()
        return result

    checkpoint = torch.load(model_dir / "best.pt", map_location="cpu", weights_only=False)
    collision_model, _ = _stage2_collision_model_from_checkpoint(checkpoint)
    collision_model.to(device).eval()

    direct_checkpoint = torch.load(
        model_dir / "direct.pt", map_location="cpu", weights_only=False
    )
    direct_model = _Stage2DirectSpatial(**direct_checkpoint["model_config"])
    direct_model.load_state_dict(direct_checkpoint["model_state_dict"])
    direct_model.to(device).eval()

    feature_config = direct_checkpoint.get("feature_config", {})
    direct_layer = feature_config.get("layer", "layer3")
    if direct_layer not in {"layer3", "layer4"}:
        raise ValueError(f"Unsupported Stage 2 spatial layer: {direct_layer}")
    short_edge = int(feature_config.get("short_edge", 256))
    if short_edge < 32 or feature_config.get("crop") is not None:
        raise ValueError("Unsupported Stage 2 Direct spatial preprocessing")
    expected_channels = 256 if direct_layer == "layer3" else 512
    if int(direct_checkpoint["model_config"]["input_channels"]) != expected_channels:
        raise ValueError("Stage 2 Direct checkpoint and feature layer do not match")
    direct_mean = torch.tensor(
        feature_config.get("mean", (0.485, 0.456, 0.406)),
        device=device,
    ).view(1, 3, 1, 1)
    direct_std = torch.tensor(
        feature_config.get("std", (0.229, 0.224, 0.225)),
        device=device,
    ).view(1, 3, 1, 1)

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
                # V0 training features were extracted in FP32, so inference uses FP32 too.
                features.append(backbone(images.to(device, non_blocking=True)).cpu())

            sequence = torch.cat(features)[None].to(device)
            collision_logits = collision_model(sequence)
            collision_index = int(collision_logits.argmax(1).item())
            del sequence, collision_logits, features

            direct_loader = DataLoader(
                _Stage2RawFrames(paths),
                batch_size=64,
                num_workers=6,
                pin_memory=True,
            )
            spatial_maps = []
            for uint8_images in direct_loader:
                images = uint8_images.to(
                    device=device, dtype=torch.float32, non_blocking=True
                ).div_(255)
                images = _stage2_resize_short_edge(images, short_edge)
                images = (images - direct_mean) / direct_std
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    maps = _stage2_spatial_features(backbone, images, direct_layer)
                spatial_maps.append(maps.half().cpu())
                del images, maps

            direct_sequence = torch.cat(spatial_maps)[None].to(
                device=device, dtype=torch.float32
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                entry_logits, side_logits, evasion_logits = direct_model(
                    direct_sequence,
                    torch.tensor([collision_index], device=device),
                )
            entry_index = int(entry_logits.argmax(1).item())
            entry_side = "RIGHT" if int(side_logits.argmax(1).item()) == 1 else "LEFT"
            evasion_space = int(evasion_logits.argmax(1).item())
            frame_numbers = [_frame_number(path) for path in paths]
            rows.append(
                {
                    "ID": folder.name,
                    "collision_frame": frame_numbers[collision_index],
                    "entry_frame": frame_numbers[entry_index],
                    "evasion_space": evasion_space,
                    "entry_side": entry_side,
                }
            )
            del direct_sequence, spatial_maps, entry_logits, side_logits, evasion_logits

    del backbone, collision_model, direct_model
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
