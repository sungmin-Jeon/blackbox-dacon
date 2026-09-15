"""Clip-consistent weak augmentation for Stage 1 training."""

from __future__ import annotations

import torch

from inference import S1_MEAN, S1_STD


AUGMENTATION_VERSION = 1
AUGMENTATION_MODES = ("none", "weak")


def augmentation_config(
    mode: str = "none",
    *,
    flip_probability: float = 0.5,
    brightness: float = 0.1,
    contrast: float = 0.1,
    saturation: float = 0.1,
    version: int = AUGMENTATION_VERSION,
) -> dict:
    """Return validated, checkpoint-friendly Stage 1 augmentation settings."""
    if mode not in AUGMENTATION_MODES:
        raise ValueError(f"Unknown augmentation mode: {mode}")
    if not 0.0 <= flip_probability <= 1.0:
        raise ValueError("augmentation flip probability must be between zero and one")
    for name, strength in (
        ("brightness", brightness),
        ("contrast", contrast),
        ("saturation", saturation),
    ):
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"augmentation {name} must be between zero and one")
    if version != AUGMENTATION_VERSION:
        raise ValueError(f"Unsupported augmentation version: {version}")
    return {
        "mode": mode,
        "flip_probability": float(flip_probability),
        "brightness": float(brightness),
        "contrast": float(contrast),
        "saturation": float(saturation),
        "version": int(version),
    }


def _random_factor(strength: float, *, device: torch.device) -> torch.Tensor:
    """Sample one scalar factor shared by every frame in a clip."""
    return 1.0 + (torch.rand((), device=device) * 2.0 - 1.0) * strength


def augment_stage1_clip(clip: torch.Tensor, augmentation: dict | None = None) -> torch.Tensor:
    """Apply one weak transform consistently to all frames of a normalized clip."""
    options = augmentation_config(**(augmentation or {}))
    if options["mode"] == "none":
        return clip
    if clip.ndim != 4 or clip.shape[0] != 3:
        raise ValueError(
            f"Stage 1 augmentation expects [3,T,H,W], got {tuple(clip.shape)}"
        )

    mean = S1_MEAN.to(device=clip.device, dtype=clip.dtype)
    std = S1_STD.to(device=clip.device, dtype=clip.dtype)
    pixels = (clip * std + mean).clamp(0.0, 1.0)

    if torch.rand((), device=clip.device) < options["flip_probability"]:
        pixels = torch.flip(pixels, dims=(-1,))

    brightness = _random_factor(options["brightness"], device=clip.device)
    pixels = pixels * brightness

    luminance_weights = pixels.new_tensor((0.2989, 0.5870, 0.1140))[:, None, None, None]
    grayscale = (pixels * luminance_weights).sum(dim=0, keepdim=True)
    contrast_center = grayscale.mean()
    contrast = _random_factor(options["contrast"], device=clip.device)
    pixels = (pixels - contrast_center) * contrast + contrast_center

    grayscale = (pixels * luminance_weights).sum(dim=0, keepdim=True)
    saturation = _random_factor(options["saturation"], device=clip.device)
    pixels = grayscale + (pixels - grayscale) * saturation
    pixels = pixels.clamp(0.0, 1.0)

    return (pixels - mean) / std
