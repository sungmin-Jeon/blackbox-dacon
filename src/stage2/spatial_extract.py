"""Extract frozen ResNet spatial maps without global average pooling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18
from tqdm.auto import tqdm


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _config_signature(config: dict) -> str:
    """Return the settings that determine the contents of a feature cache."""
    keys = (
        "version",
        "backbone",
        "backbone_path",
        "layer",
        "channels",
        "short_edge",
        "crop",
        "mean",
        "std",
        "frame_number_base",
        "storage_dtype",
    )
    return json.dumps(
        {key: config.get(key) for key in keys},
        sort_keys=True,
        separators=(",", ":"),
    )


class ResNet18Spatial(nn.Module):
    def __init__(self, backbone: nn.Module, layer: str) -> None:
        super().__init__()
        if layer not in {"layer3", "layer4"}:
            raise ValueError("layer must be layer3 or layer4")
        children = [
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3,
        ]
        if layer == "layer4":
            children.append(backbone.layer4)
        self.features = nn.Sequential(*children)

    def forward(self, images: Tensor) -> Tensor:
        return self.features(images)


def resize_short_edge(images: Tensor, short_edge: int) -> Tensor:
    if images.ndim != 4 or short_edge < 32:
        raise ValueError("images must be [B,C,H,W] and short_edge must be >= 32")
    height, width = images.shape[-2:]
    scale = short_edge / min(height, width)
    output_height = max(1, round(height * scale))
    output_width = max(1, round(width * scale))
    return F.interpolate(
        images,
        size=(output_height, output_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )


def _load_backbone(weights_path: Path | None) -> nn.Module:
    if weights_path is None:
        return resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    backbone = resnet18(weights=None)
    backbone.load_state_dict(checkpoint)
    return backbone


def _video_index(root: Path) -> dict[str, Path]:
    paths = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    index: dict[str, Path] = {}
    duplicates = []
    for path in paths:
        if path.stem in index:
            duplicates.append(path.stem)
        index[path.stem] = path
    if duplicates:
        raise ValueError(f"Duplicate video IDs under {root}: {sorted(set(duplicates))[:10]}")
    return index


def _read_batches(path: Path, batch_size: int):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        capture.release()
        raise RuntimeError(f"Invalid FPS: {path}")
    frames = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(rgb).permute(2, 0, 1))
        if len(frames) == batch_size:
            yield torch.stack(frames), fps
            frames.clear()
    capture.release()
    if frames:
        yield torch.stack(frames), fps


def extract_video(
    path: Path,
    model: nn.Module,
    device: torch.device,
    *,
    batch_size: int,
    short_edge: int,
    amp: bool,
) -> tuple[Tensor, float]:
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    batches = []
    fps = 0.0
    with torch.inference_mode():
        for uint8_frames, fps in _read_batches(path, batch_size):
            images = uint8_frames.to(device=device, dtype=torch.float32).div_(255)
            images = resize_short_edge(images, short_edge)
            images = (images - mean) / std
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp and device.type == "cuda",
            ):
                batches.append(model(images).half().cpu())
    if not batches:
        raise RuntimeError(f"No decodable frames: {path}")
    return torch.cat(batches), fps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True, help="Direct-label CSV with ID column")
    parser.add_argument("--video-root", type=Path, required=True, help="Folder searched recursively for videos")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, default=None, help="Optional official ResNet18 state dict")
    parser.add_argument("--layer", choices=("layer3", "layer4"), default="layer3")
    parser.add_argument("--short-edge", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--frame-number-base", type=int, choices=(0, 1), default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true", help="Replace existing per-video caches")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    labels = pd.read_csv(args.labels, dtype={"ID": str})
    if "ID" not in labels:
        raise ValueError("labels CSV requires an ID column")
    if "status" in labels:
        labels = labels[labels["status"].fillna("").str.upper().eq("KEEP")]
    ids = labels["ID"].dropna().astype(str).drop_duplicates().tolist()
    if not ids:
        raise ValueError("No labelled video IDs found")

    video_root = args.video_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    index = _video_index(video_root)
    missing = [video_id for video_id in ids if video_id not in index]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} labelled videos, for example: {missing[:10]}")

    device = torch.device(args.device)
    backbone_path = args.backbone.expanduser().resolve() if args.backbone else None
    model = ResNet18Spatial(_load_backbone(backbone_path), args.layer).to(device).eval()
    model.requires_grad_(False)
    channels = 256 if args.layer == "layer3" else 512
    config = {
        "version": 1,
        "backbone": "resnet18_imagenet1k_v1",
        "backbone_path": str(backbone_path) if backbone_path else None,
        "layer": args.layer,
        "channels": channels,
        "short_edge": args.short_edge,
        "crop": None,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "frame_number_base": args.frame_number_base,
        "storage_dtype": "float16",
    }
    config_path = output_dir / "feature_config.json"
    existing_caches = list(output_dir.glob("*.pt"))
    if config_path.is_file() and existing_caches and not args.force:
        previous_config = json.loads(config_path.read_text(encoding="utf-8"))
        if _config_signature(previous_config) != _config_signature(config):
            raise ValueError(
                "Existing feature cache uses different extraction settings. "
                "Choose another output directory or rerun with --force."
            )
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    rows = []
    for video_id in tqdm(ids, desc=f"Extracting {args.layer} maps"):
        target = output_dir / f"{video_id}.pt"
        if target.exists() and not args.force:
            cached = torch.load(target, map_location="cpu", weights_only=True)
            cached_config = cached.get("feature_config", {})
            if _config_signature(cached_config) != _config_signature(config):
                raise ValueError(
                    f"Cached settings do not match this run: {target}. "
                    "Choose another output directory or rerun with --force."
                )
            maps = cached["features"]
            fps = float(cached["fps"])
        else:
            maps, fps = extract_video(
                index[video_id], model, device,
                batch_size=args.batch_size,
                short_edge=args.short_edge,
                amp=not args.no_amp,
            )
            frame_numbers = torch.arange(
                args.frame_number_base,
                args.frame_number_base + len(maps),
                dtype=torch.int32,
            )
            temporary = target.with_suffix(".tmp")
            torch.save(
                {
                    "ID": video_id,
                    "features": maps,
                    "frame_numbers": frame_numbers,
                    "fps": fps,
                    "feature_config": config,
                },
                temporary,
            )
            temporary.replace(target)
        rows.append({
            "ID": video_id,
            "video_path": str(index[video_id]),
            "feature_path": str(target),
            "fps": fps,
            "frames": int(len(maps)),
            "channels": int(maps.shape[1]),
            "height": int(maps.shape[2]),
            "width": int(maps.shape[3]),
        })

    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "manifest.csv", index=False, encoding="utf-8-sig")
    gib = sum(path.stat().st_size for path in output_dir.glob("*.pt")) / 1024**3
    print(f"Saved {len(manifest)} videos to {output_dir}")
    print(f"Cache size: {gib:.2f} GiB")
    print(manifest[["frames", "channels", "height", "width"]].describe().T)


if __name__ == "__main__":
    main()
