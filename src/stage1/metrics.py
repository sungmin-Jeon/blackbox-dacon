"""Metrics used by the Stage 1 binary classifier."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    macro_f1: float
    original_f1: float
    rerecorded_f1: float
    predicted_original_ratio: float
    predicted_rerecorded_ratio: float


def _class_f1(targets: list[int], predictions: list[int], class_id: int) -> float:
    true_positive = sum(
        target == class_id and prediction == class_id
        for target, prediction in zip(targets, predictions)
    )
    false_positive = sum(
        target != class_id and prediction == class_id
        for target, prediction in zip(targets, predictions)
    )
    false_negative = sum(
        target == class_id and prediction != class_id
        for target, prediction in zip(targets, predictions)
    )

    denominator = 2 * true_positive + false_positive + false_negative
    if denominator == 0:
        return 0.0
    return 2 * true_positive / denominator


def classification_metrics(
    targets: list[int],
    predictions: list[int],
) -> ClassificationMetrics:
    """Compute binary accuracy, Macro-F1, per-class F1, and prediction ratios."""
    if not targets:
        raise ValueError("targets cannot be empty")
    if len(targets) != len(predictions):
        raise ValueError("targets and predictions must have the same length")

    allowed_labels = {0, 1}
    if not set(targets).issubset(allowed_labels):
        raise ValueError("targets must contain only class IDs 0 and 1")
    if not set(predictions).issubset(allowed_labels):
        raise ValueError("predictions must contain only class IDs 0 and 1")

    total = len(targets)
    correct = sum(target == prediction for target, prediction in zip(targets, predictions))
    original_f1 = _class_f1(targets, predictions, class_id=0)
    rerecorded_f1 = _class_f1(targets, predictions, class_id=1)

    return ClassificationMetrics(
        accuracy=correct / total,
        macro_f1=(original_f1 + rerecorded_f1) / 2,
        original_f1=original_f1,
        rerecorded_f1=rerecorded_f1,
        predicted_original_ratio=predictions.count(0) / total,
        predicted_rerecorded_ratio=predictions.count(1) / total,
    )
