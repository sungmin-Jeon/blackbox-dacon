"""Checks Entry feature loading and paired temporal-prior metrics."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from src.stage2.entry_feature_compare import entry_metrics, load_feature_samples


class EntryFeatureCompareTests(unittest.TestCase):
    def test_loader_preserves_entry_target_prior_and_confidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save(
                {
                    "features": torch.randn(4, 256, 3, 5),
                    "frame_numbers": torch.arange(1, 5),
                    "fps": 15.0,
                },
                root / "sample.pt",
            )
            labels = pd.DataFrame(
                [
                    {
                        "ID": "sample",
                        "split": "val",
                        "entry_frame": 3,
                        "entry_confidence": "high",
                    }
                ]
            )
            samples, info = load_feature_samples(labels, root, {"sample": 2})
            self.assertEqual(info["feature_kind"], "spatial")
            self.assertEqual(samples[0]["target_index"], 2)
            self.assertEqual(samples[0]["prior_frame"], 2)
            self.assertEqual(samples[0]["entry_confidence"], "HIGH")

    def test_metrics_separate_model_and_prior_wins(self):
        predictions = pd.DataFrame(
            {
                "error_frames": [0, 1, 10, 10],
                "abs_error_sec": [0.0, 0.1, 1.0, 1.0],
                "model_correct_03": [True, True, False, False],
                "prior_correct_03": [True, False, True, False],
                "entry_confidence": ["HIGH", "MEDIUM", "LOW", "HIGH"],
            }
        )
        metrics = entry_metrics(predictions)
        self.assertEqual(metrics["acc_03"], 0.5)
        self.assertEqual(metrics["prior_acc_03"], 0.5)
        self.assertEqual(metrics["both_correct"], 1)
        self.assertEqual(metrics["model_only"], 1)
        self.assertEqual(metrics["prior_only"], 1)
        self.assertEqual(metrics["both_wrong"], 1)
        self.assertEqual(metrics["medium_acc_03"], 1.0)
        self.assertEqual(metrics["low_acc_03"], 0.0)


if __name__ == "__main__":
    unittest.main()
