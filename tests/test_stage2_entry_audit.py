"""Checks Stage 2 entry-label auditing and content-free baselines."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.stage2.entry_audit import (
    label_summary,
    load_entry_labels,
    prior_metrics,
    temporal_prior_predictions,
)


class EntryAuditTests(unittest.TestCase):
    def test_audit_validates_labels_and_scores_priors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            pd.DataFrame(
                [
                    {
                        "ID": f"v{index}",
                        "split": "train" if index < 3 else "val",
                        "status": "KEEP",
                        "fps": 10,
                        "total_frames": 100,
                        "entry_frame": frame,
                        "entry_min": frame - 1 if frame > 1 else 1,
                        "entry_max": frame + 1,
                        "collision_frame": frame + 10,
                        "entry_confidence": "HIGH",
                        "entry_side": "LEFT" if index % 2 else "RIGHT",
                    }
                    for index, frame in enumerate((20, 30, 40, 30, 31))
                ]
            ).to_csv(path, index=False)

            labels = load_entry_labels(path)
            predictions, priors = temporal_prior_predictions(labels)
            metrics = prior_metrics(predictions)
            summary = label_summary(labels)

            self.assertEqual(len(labels), 5)
            self.assertEqual(priors["train_median_frame"], 30.0)
            self.assertEqual(set(predictions["method"]), {
                "median_frame", "median_time", "median_ratio"
            })
            self.assertTrue((metrics["acc_03"] == 1.0).all())
            self.assertEqual(summary["entry_interval_available"], 5)
            self.assertEqual(summary["entry_after_collision_count"], 0)

    def test_bad_entry_interval_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            pd.DataFrame(
                [
                    {
                        "ID": "train",
                        "split": "train",
                        "fps": 15,
                        "total_frames": 100,
                        "entry_frame": 20,
                        "entry_min": 21,
                        "entry_max": 22,
                    },
                    {
                        "ID": "val",
                        "split": "val",
                        "fps": 15,
                        "total_frames": 100,
                        "entry_frame": 20,
                        "entry_min": 19,
                        "entry_max": 21,
                    },
                ]
            ).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Invalid entry interval"):
                load_entry_labels(path)


if __name__ == "__main__":
    unittest.main()
