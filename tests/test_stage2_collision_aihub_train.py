"""Checks the AIHub-only collision training utilities."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from src.stage2.collision_aihub_train import (
    baseline_metrics_on_validation,
    collision_metrics,
    load_collision_labels,
    load_collision_sample,
)
from src.stage2.collision_predict import collision_model_from_checkpoint


class AIHubCollisionTrainingTests(unittest.TestCase):
    def test_labels_keep_only_annotated_train_and_val_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            pd.DataFrame(
                [
                    {"ID": "a", "split": "train", "collision_frame": 2, "status": "KEEP"},
                    {"ID": "b", "split": "val", "collision_frame": 3, "status": "KEEP"},
                    {"ID": "c", "split": "train", "collision_frame": None, "status": "KEEP"},
                    {"ID": "d", "split": "val", "collision_frame": 2, "status": "SKIP"},
                ]
            ).to_csv(path, index=False)
            labels = load_collision_labels(path)
            self.assertEqual(labels["ID"].tolist(), ["a", "b"])
            self.assertEqual(labels["collision_frame"].tolist(), [2, 3])

    def test_cache_sample_and_checkpoint_are_submission_compatible(self):
        from src.stage2.collision_predict import CollisionBiGRU

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save(
                {
                    "features": torch.randn(4, 512),
                    "frame_numbers": torch.arange(1, 5),
                    "fps": 15.0,
                },
                root / "sample.pt",
            )
            row = next(
                pd.DataFrame(
                    [{"ID": "sample", "split": "train", "collision_frame": 3}]
                ).itertuples(index=False)
            )
            sample = load_collision_sample(row, root)
            self.assertEqual(sample["target_index"], 2)

            original = CollisionBiGRU(512, 8, 1, 0.3).eval()
            restored, checkpoint_format = collision_model_from_checkpoint(
                {
                    "model_state_dict": original.state_dict(),
                    "dropout": 0.3,
                    "task": "collision_frame_aihub_only",
                }
            )
            self.assertEqual(checkpoint_format, "collision_v0")
            features = sample["x"].unsqueeze(0)
            restored.eval()
            torch.testing.assert_close(original(features), restored(features))

    def test_baseline_is_scored_on_the_same_validation_ids(self):
        samples = [
            {"ID": "a", "target_frame": 10, "fps": 10.0},
            {"ID": "b", "target_frame": 20, "fps": 10.0},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            pd.DataFrame(
                {"ID": ["b", "a", "extra"], "collision_pred_frame": [24, 11, 1]}
            ).to_csv(path, index=False)
            metrics, rows = baseline_metrics_on_validation(path, samples)
            self.assertEqual(rows["ID"].tolist(), ["a", "b"])
            self.assertAlmostEqual(metrics["acc_03"], 0.5)
            self.assertAlmostEqual(metrics["median_sec"], 0.25)

    def test_collision_metrics_match_time_threshold(self):
        metrics = collision_metrics(
            pd.DataFrame(
                {
                    "error_frames": [0, 3, -4],
                    "abs_error_sec": [0.0, 0.3, 0.4],
                }
            )
        )
        self.assertAlmostEqual(metrics["exact"], 1 / 3)
        self.assertAlmostEqual(metrics["acc_03"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
