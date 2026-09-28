"""Audit Stage 2 entry labels and measure content-free temporal priors."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return result


def _integer(value: Any, name: str) -> int:
    result = _number(value, name)
    if not result.is_integer():
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return int(result)


def load_entry_labels(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(path, dtype={"ID": str})
    if "entry_frame" not in labels and "entry_best" in labels:
        labels["entry_frame"] = labels["entry_best"]
    required = {"ID", "split", "entry_frame", "fps", "total_frames"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Missing entry audit columns: {sorted(missing)}")
    if "status" in labels:
        labels = labels[labels["status"].fillna("").str.upper().eq("KEEP")]
    labels = labels.copy()
    labels["split"] = labels["split"].astype(str).str.lower()
    labels = labels[labels["split"].isin({"train", "val"})]
    if labels["ID"].duplicated().any():
        duplicates = labels.loc[labels["ID"].duplicated(False), "ID"].tolist()
        raise ValueError(f"Duplicate IDs: {duplicates[:10]}")
    if not {"train", "val"}.issubset(set(labels["split"])):
        raise ValueError("Both train and val splits are required")

    for index, row in labels.iterrows():
        entry = _integer(row["entry_frame"], "entry_frame")
        total = _integer(row["total_frames"], "total_frames")
        fps = _number(row["fps"], "fps")
        if total < 1 or not 1 <= entry <= total or fps <= 0:
            raise ValueError(
                f"Invalid entry/fps/total for {row['ID']}: {entry}, {fps}, {total}"
            )
        labels.at[index, "entry_frame"] = entry
        labels.at[index, "total_frames"] = total
        labels.at[index, "fps"] = fps

        has_min = "entry_min" in labels and pd.notna(row.get("entry_min"))
        has_max = "entry_max" in labels and pd.notna(row.get("entry_max"))
        if has_min != has_max:
            raise ValueError(f"entry_min/max must both exist for {row['ID']}")
        if has_min:
            low = _integer(row["entry_min"], "entry_min")
            high = _integer(row["entry_max"], "entry_max")
            if not 1 <= low <= entry <= high <= total:
                raise ValueError(
                    f"Invalid entry interval for {row['ID']}: {low}, {entry}, {high}"
                )
            labels.at[index, "entry_min"] = low
            labels.at[index, "entry_max"] = high

    labels["entry_frame"] = labels["entry_frame"].astype(int)
    labels["total_frames"] = labels["total_frames"].astype(int)
    labels["fps"] = labels["fps"].astype(float)
    labels["entry_time_sec"] = (labels["entry_frame"] - 1) / labels["fps"]
    labels["entry_ratio"] = np.where(
        labels["total_frames"] > 1,
        (labels["entry_frame"] - 1) / (labels["total_frames"] - 1),
        0.0,
    )
    return labels.reset_index(drop=True)


def _clip_frame(values: pd.Series, totals: pd.Series) -> pd.Series:
    return pd.Series(
        np.maximum(1, np.minimum(np.rint(values).astype(int), totals.astype(int))),
        index=values.index,
        dtype=int,
    )


def temporal_prior_predictions(
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    train = labels[labels["split"] == "train"]
    val = labels[labels["split"] == "val"].copy()
    priors = {
        "train_median_frame": float(train["entry_frame"].median()),
        "train_median_time_sec": float(train["entry_time_sec"].median()),
        "train_median_ratio": float(train["entry_ratio"].median()),
    }
    predictions = []
    candidates = {
        "median_frame": pd.Series(
            priors["train_median_frame"], index=val.index, dtype=float
        ),
        "median_time":
            priors["train_median_time_sec"] * val["fps"] + 1,
        "median_ratio":
            priors["train_median_ratio"] * (val["total_frames"] - 1) + 1,
    }
    for method, candidate in candidates.items():
        predicted = _clip_frame(candidate, val["total_frames"])
        error_frames = predicted - val["entry_frame"]
        predictions.append(
            pd.DataFrame(
                {
                    "ID": val["ID"],
                    "method": method,
                    "target_frame": val["entry_frame"],
                    "predicted_frame": predicted,
                    "fps": val["fps"],
                    "error_frames": error_frames,
                    "abs_error_sec": error_frames.abs() / val["fps"],
                }
            )
        )
    return pd.concat(predictions, ignore_index=True), priors


def prior_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, frame in predictions.groupby("method", sort=False):
        errors = frame["abs_error_sec"]
        rows.append(
            {
                "method": method,
                "videos": len(frame),
                "acc_03": float((errors <= 0.3 + 1e-9).mean()),
                "exact": float((frame["error_frames"] == 0).mean()),
                "median_sec": float(errors.median()),
                "mean_sec": float(errors.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["acc_03", "median_sec", "mean_sec"], ascending=[False, True, True]
    )


def label_summary(labels: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "videos": len(labels),
        "split_counts": labels["split"].value_counts().to_dict(),
        "first_frame_counts": {
            split: int((frame["entry_frame"] == 1).sum())
            for split, frame in labels.groupby("split")
        },
        "entry_frame": labels["entry_frame"].describe().to_dict(),
        "entry_time_sec": labels["entry_time_sec"].describe().to_dict(),
        "entry_ratio": labels["entry_ratio"].describe().to_dict(),
    }
    for column in ("entry_confidence", "entry_side", "lane_basis"):
        if column in labels:
            summary[f"{column}_counts"] = (
                labels[column].fillna("MISSING").astype(str).value_counts().to_dict()
            )
    if {"entry_min", "entry_max"}.issubset(labels.columns):
        bounded = labels.dropna(subset=["entry_min", "entry_max"]).copy()
        if len(bounded):
            widths = (bounded["entry_max"] - bounded["entry_min"]) / bounded["fps"]
            summary["entry_interval_available"] = len(bounded)
            summary["entry_interval_width_sec"] = widths.describe().to_dict()
    if "collision_frame" in labels:
        collision = pd.to_numeric(labels["collision_frame"], errors="coerce")
        available = collision.notna()
        gaps = (collision[available] - labels.loc[available, "entry_frame"]) / labels.loc[
            available, "fps"
        ]
        summary["collision_available"] = int(available.sum())
        if len(gaps):
            summary["entry_to_collision_sec"] = gaps.describe().to_dict()
            summary["entry_after_collision_count"] = int((gaps < 0).sum())
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = load_entry_labels(args.labels.expanduser().resolve())
    if args.expected_count is not None and len(labels) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} labels, got {len(labels)}")
    predictions, priors = temporal_prior_predictions(labels)
    metrics = prior_metrics(predictions)
    summary = label_summary(labels)
    summary["temporal_priors"] = priors

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    labels.to_csv(output_dir / "entry_audit_rows.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(
        output_dir / "entry_prior_predictions.csv", index=False, encoding="utf-8-sig"
    )
    metrics.to_csv(
        output_dir / "entry_prior_metrics.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "entry_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Labels: {len(labels)} | Split: {summary['split_counts']}")
    print(f"First-frame entry labels: {summary['first_frame_counts']}")
    if "entry_confidence_counts" in summary:
        print(f"Entry confidence: {summary['entry_confidence_counts']}")
    if "entry_interval_available" in summary:
        interval = summary["entry_interval_width_sec"]
        print(
            "Entry intervals: "
            f"{summary['entry_interval_available']} | "
            f"median width {interval['50%']:.3f}s | max {interval['max']:.3f}s"
        )
    print("Content-free validation baselines")
    print(metrics.to_string(index=False))
    print(f"Saved: {output_dir}")


if __name__ == "__main__":
    main()
