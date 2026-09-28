"""Selectable Stage 2 losses, also usable as a single Colab code cell.

Temporal API (the current notebook trains one whole video at a time)::

    train_loss_fn = make_temporal_loss("gaussian_ce", sigma_sec=0.1)
    loss = train_loss_fn(logits, sample)

``logits`` is [T] or [1, T], without padding. ``sample`` contains ``frames``
(actual frame numbers), a zero-based ``target_index``, and ``fps`` for constant
frame-rate video OR ``frame_times`` (seconds). Annotation losses additionally
require ``target_min_frame`` and ``target_max_frame``. Missing targets must be
excluded by the training loop; they are never replaced with frame zero.

All temporal choices keep the existing per-frame head and argmax decoding.
Regression losses are separate: they need differentiable predicted times, not
argmax frame indices. Keep validation hard CE / Accuracy@0.3s unchanged when
comparing training losses; raw losses from different families are not comparable.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _number(value: Any, name: str, *, minimum: float = 0.0, strict: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result) or result < minimum or (strict and result == minimum):
        op = ">" if strict else ">="
        raise ValueError(f"{name} must be finite and {op} {minimum}")
    return result


def _index(value: Any, name: str) -> int:
    number = _number(value, name)
    if not number.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(number)


@dataclass
class _Temporal:
    scores: Tensor
    logp: Tensor
    p: Tensor
    frames: Tensor
    times: Tensor
    delta: Tensor
    target: int
    sample: Mapping[str, Any]


def _prepare(logits: Tensor, sample: Mapping[str, Any]) -> _Temporal:
    if logits.ndim == 2 and logits.shape[0] == 1:
        logits = logits[0]
    if logits.ndim != 1 or logits.numel() == 0 or not logits.is_floating_point():
        raise ValueError("Temporal logits must be floating [T] or [1, T], one unpadded video")
    scores = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    if not torch.isfinite(scores).all():
        raise ValueError("Temporal logits contain NaN/Inf")
    target = _index(sample.get("target_index"), "target_index")
    if target >= len(scores):
        raise ValueError("target_index is outside this video")
    if "frames" not in sample:
        raise ValueError("sample['frames'] must contain actual frame numbers")
    frames = torch.as_tensor(sample["frames"], device=scores.device, dtype=torch.float64)
    if frames.shape != scores.shape or not torch.isfinite(frames).all():
        raise ValueError("frames must be finite and have shape [T]")
    if (frames < 0).any() or (frames != frames.round()).any() or not (frames[1:] > frames[:-1]).all():
        raise ValueError("frames must be nonnegative, strictly increasing integer frame numbers")
    if "target_frame" in sample and _index(sample["target_frame"], "target_frame") != frames[target].item():
        raise ValueError("target_frame does not match frames[target_index]")
    if "frame_times" in sample:
        times = torch.as_tensor(sample["frame_times"], device=scores.device, dtype=torch.float64)
    else:
        fps = _number(sample.get("fps"), "fps", strict=True)
        times = (frames - frames[0]) / fps
    if times.shape != scores.shape or not torch.isfinite(times).all():
        raise ValueError("frame_times must be finite seconds with shape [T]")
    if not (times[1:] > times[:-1]).all():
        raise ValueError("frame_times must be strictly increasing")
    logp = F.log_softmax(scores, dim=0)
    return _Temporal(scores, logp, logp.exp(), frames, times, times - times[target], target, sample)


def _window(c: _Temporal, radius_sec: float) -> Tensor:
    radius = _number(radius_sec, "radius_sec")
    return c.delta.abs() <= radius + 1e-9


def _annotation(c: _Temporal) -> Tensor:
    low = _index(c.sample.get("target_min_frame"), "target_min_frame")
    high = _index(c.sample.get("target_max_frame"), "target_max_frame")
    if not low <= c.frames[c.target].item() <= high:
        raise ValueError("Annotation bounds must satisfy min <= target frame <= max")
    return (c.frames >= low) & (c.frames <= high)


def _gaussian(c: _Temporal, sigma_sec: float) -> Tensor:
    sigma = _number(sigma_sec, "sigma_sec", strict=True)
    # Subtract the largest exponent implicitly with softmax; target has delta=0.
    return torch.softmax(-0.5 * (c.delta / sigma).square(), dim=0).to(c.scores.dtype)


def _soft_ce(c: _Temporal, q: Tensor) -> Tensor:
    return -(q * c.logp).sum()


def _hard_ce(c: _Temporal) -> Tensor:
    return -c.logp[c.target]


def _gaussian_ce(c: _Temporal, sigma_sec: float = 0.1) -> Tensor:
    return _soft_ce(c, _gaussian(c, sigma_sec))


def _triangular_ce(c: _Temporal, radius_sec: float = 0.3) -> Tensor:
    radius = _number(radius_sec, "radius_sec", strict=True)
    q = (1 - c.delta.abs() / radius).clamp_min(0).to(c.scores.dtype)
    return _soft_ce(c, q / q.sum())


def _window_ce(c: _Temporal, radius_sec: float = 0.3) -> Tensor:
    q = _window(c, radius_sec).to(c.scores.dtype)
    return _soft_ce(c, q / q.sum())


def _annotation_ce(c: _Temporal) -> Tensor:
    q = _annotation(c).to(c.scores.dtype)
    return _soft_ce(c, q / q.sum())


def _label_smoothing(c: _Temporal, smoothing: float = 0.1) -> Tensor:
    epsilon = _number(smoothing, "smoothing")
    if epsilon > 1:
        raise ValueError("smoothing must be <= 1")
    return -(1 - epsilon) * c.logp[c.target] - epsilon * c.logp.mean()


def _window_nll(c: _Temporal, radius_sec: float = 0.3) -> Tensor:
    # Reward total probability in the window, not a uniform distribution in it.
    return -torch.logsumexp(c.logp[_window(c, radius_sec)], dim=0)


def _annotation_nll(c: _Temporal) -> Tensor:
    return -torch.logsumexp(c.logp[_annotation(c)], dim=0)


def _gaussian_kl(c: _Temporal, sigma_sec: float = 0.1) -> Tensor:
    # Same gradient as gaussian_ce for the same fixed target and reduction.
    return F.kl_div(c.logp, _gaussian(c, sigma_sec), reduction="sum")


def _wasserstein1(c: _Temporal) -> Tensor:
    # Exact W1 against a point target: expected absolute time error, in seconds.
    return (c.p * c.delta.abs().to(c.scores.dtype)).sum()


def _cdf_l2(c: _Temporal) -> Tensor:
    # Integral of squared CDF error, not the squared Wasserstein-2 distance.
    target_cdf = (torch.arange(len(c.p), device=c.p.device) >= c.target).to(c.p.dtype)
    widths = (c.times[1:] - c.times[:-1]).to(c.p.dtype)
    return ((c.p.cumsum(0) - target_cdf)[:-1].square() * widths).sum()


def _bce(c: _Temporal, radius_sec: float = 0.0, pos_weight: float | str = 1.0) -> Tensor:
    target = _window(c, radius_sec).to(c.scores.dtype)
    if pos_weight == "balanced":
        weight = ((len(target) - target.sum()) / target.sum()).clamp_min(1.0)
    else:
        weight = c.scores.new_tensor(_number(pos_weight, "pos_weight", strict=True))
    return F.binary_cross_entropy_with_logits(c.scores, target, pos_weight=weight)


def _focal_ce(c: _Temporal, gamma: float = 2.0) -> Tensor:
    gamma = _number(gamma, "gamma")
    return -(1 - c.p[c.target]).pow(gamma) * c.logp[c.target]


def _focal_bce(c: _Temporal, gamma: float = 2.0, alpha: float | None = None,
               radius_sec: float = 0.0) -> Tensor:
    gamma = _number(gamma, "gamma")
    target = _window(c, radius_sec).to(c.scores.dtype)
    ce = F.binary_cross_entropy_with_logits(c.scores, target, reduction="none")
    loss = (1 - (-ce).exp()).pow(gamma) * ce
    if alpha is not None:
        alpha = _number(alpha, "alpha")
        if alpha > 1:
            raise ValueError("alpha must be <= 1")
        loss = loss * (target * alpha + (1 - target) * (1 - alpha))
    return loss.mean()


def _heatmap_mse(c: _Temporal, sigma_sec: float = 0.1) -> Tensor:
    sigma = _number(sigma_sec, "sigma_sec", strict=True)
    # Peak is 1; this is NOT the unit-sum target used by gaussian_ce.
    heatmap = (-0.5 * (c.delta / sigma).square()).exp().to(c.scores.dtype)
    return F.mse_loss(c.scores.sigmoid(), heatmap)


def _ranking(c: _Temporal, margin: float = 1.0, radius_sec: float = 0.0) -> Tensor:
    margin = _number(margin, "margin")
    negatives = c.scores[~_window(c, radius_sec)]
    if negatives.numel() == 0:
        return c.scores.sum() * 0
    return F.relu(margin - c.scores[c.target] + negatives).mean()


def _ordinal_bce(c: _Temporal) -> Tensor:
    # P(event has occurred by frame k) = cumulative event probability.
    # Omit the last threshold: its cumulative probability is identically 1.
    if len(c.scores) == 1:
        return c.scores.sum() * 0
    log_cdf = torch.logcumsumexp(c.logp, dim=0)[:-1]
    log_survival = torch.logcumsumexp(c.logp.flip(0), dim=0).flip(0)[1:]
    occurred = torch.arange(len(log_cdf), device=c.p.device) >= c.target
    return -torch.where(occurred, log_cdf, log_survival).mean()


_TEMPORAL_LOSSES = {
    "hard_ce": _hard_ce,
    "gaussian_ce": _gaussian_ce,
    "triangular_ce": _triangular_ce,
    "window_ce": _window_ce,
    "annotation_ce": _annotation_ce,
    "label_smoothing": _label_smoothing,
    "window_nll": _window_nll,
    "annotation_nll": _annotation_nll,
    "gaussian_kl": _gaussian_kl,
    "wasserstein1": _wasserstein1,
    "cdf_l2": _cdf_l2,
    "bce": _bce,
    "focal_ce": _focal_ce,
    "focal_bce": _focal_bce,
    "heatmap_mse": _heatmap_mse,
    "ranking": _ranking,
    "ordinal_bce": _ordinal_bce,
}
TEMPORAL_LOSS_NAMES = tuple(_TEMPORAL_LOSSES)


class TemporalLoss(nn.Module):
    """One unpadded video per call. Save ``.config`` with each checkpoint."""

    def __init__(self, name: str, **kwargs: Any):
        super().__init__()
        if name not in _TEMPORAL_LOSSES:
            raise ValueError(f"Unknown temporal loss {name!r}; choose from {TEMPORAL_LOSS_NAMES}")
        self.name = name
        self.function = _TEMPORAL_LOSSES[name]
        arguments = inspect.signature(self.function).bind(None, **kwargs)
        arguments.apply_defaults()
        self.options = {key: value for key, value in arguments.arguments.items() if key != "c"}

    @property
    def config(self) -> dict[str, Any]:
        return {"name": self.name, **self.options}

    def forward(self, logits: Tensor, sample: Mapping[str, Any]) -> Tensor:
        return self.function(_prepare(logits, sample), **self.options)


def make_temporal_loss(name: str = "hard_ce", **kwargs: Any) -> TemporalLoss:
    return TemporalLoss(name, **kwargs)


class _FocalClassification(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = _number(gamma, "gamma")

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        ce = F.cross_entropy(logits, target, reduction="none")
        return ((1 - (-ce).exp()).pow(self.gamma) * ce).mean()


class _SoftMacroF1(nn.Module):
    """Surrogate on a batch, NOT the official thresholded Macro-F1 metric."""

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        p = logits.softmax(dim=-1)
        y = F.one_hot(target, num_classes=logits.shape[-1]).to(p.dtype)
        tp = (p * y).sum(dim=0)
        return 1 - (2 * tp / (p.sum(dim=0) + y.sum(dim=0)).clamp_min(1e-8)).mean()


def make_classification_loss(name: str = "ce", **kwargs: Any) -> nn.Module:
    """Side/evasion: ce/focal/soft_macro_f1 use [B,C]; bce uses [B] or [B,1].

    Filter missing targets before calling. Standard CE accepts ``weight`` and
    ``label_smoothing``; BCE accepts ``pos_weight``. Tensor weights belong on
    the same device as logits (call loss_fn.to(device)). Soft Macro-F1 needs
    a representative batch and is not a recommended one-video objective.
    """
    factories = {
        "ce": nn.CrossEntropyLoss,
        "bce": nn.BCEWithLogitsLoss,
        "focal": _FocalClassification,
        "soft_macro_f1": _SoftMacroF1,
    }
    if name not in factories:
        raise ValueError(f"Unknown classification loss {name!r}; choose from {tuple(factories)}")
    return factories[name](**kwargs)


def make_regression_loss(name: str = "huber", **kwargs: Any) -> nn.Module:
    """Differentiable predicted times vs target times, in the SAME units/shape.

    This is not a drop-in replacement for temporal_loss(logits, sample).
    Never regress on argmax indices: argmax breaks that gradient path.
    """
    factories = {"l1": nn.L1Loss, "mse": nn.MSELoss,
                 "huber": nn.HuberLoss, "smooth_l1": nn.SmoothL1Loss}
    if name not in factories:
        raise ValueError(f"Unknown regression loss {name!r}; choose from {tuple(factories)}")
    return factories[name](**kwargs)
