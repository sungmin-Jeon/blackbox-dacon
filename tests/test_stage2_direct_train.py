"""Checks for Stage 2 Direct training behavior."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.stage2.direct_train import _combined_loss, _context_index, _load_labels


class DirectTrainingTests(unittest.TestCase):
    def test_collision_context_sources_and_mixture(self):
        sample = {
            "collision_index": 2,
            "collision_pred_index": 7,
            "maps": torch.zeros(10, 2, 2, 2),
        }
        self.assertEqual(
            _context_index(sample, source="annotated", jitter=0, rng=None), 2
        )
        self.assertEqual(
            _context_index(sample, source="predicted", jitter=0, rng=None), 7
        )
        self.assertEqual(
            _context_index(
                sample,
                source="mixed",
                jitter=0,
                rng=np.random.default_rng(1),
                ground_truth_probability=1.0,
            ),
            2,
        )
        self.assertEqual(
            _context_index(
                sample,
                source="mixed",
                jitter=0,
                rng=np.random.default_rng(1),
                ground_truth_probability=0.0,
            ),
            7,
        )
        sample["collision_index"] = None
        self.assertEqual(
            _context_index(
                sample,
                source="mixed",
                jitter=0,
                rng=np.random.default_rng(1),
            ),
            7,
        )

    def test_collision_predictions_are_merged_and_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels_path = root / "labels.csv"
            predictions_path = root / "predictions.csv"
            pd.DataFrame(
                [
                    {"ID": "train", "split": "train", "entry_side": "LEFT", "evasion_space": 0,
                     "entry_frame": 2, "collision_frame": 3},
                    {"ID": "val", "split": "val", "entry_side": "RIGHT", "evasion_space": 1,
                     "entry_frame": 2, "collision_frame": np.nan},
                ]
            ).to_csv(labels_path, index=False)
            pd.DataFrame(
                [
                    {"ID": "train", "collision_pred_frame": 4},
                    {"ID": "val", "collision_pred_frame": 5},
                ]
            ).to_csv(predictions_path, index=False)
            labels = _load_labels(labels_path, predictions_path)
            self.assertEqual(labels["collision_pred_frame"].tolist(), [4, 5])

            pd.DataFrame(
                [{"ID": "train", "collision_pred_frame": 4}]
            ).to_csv(predictions_path, index=False)
            with self.assertRaises(ValueError):
                _load_labels(labels_path, predictions_path)

    def test_single_sample_class_weight_is_not_cancelled(self):
        outputs = {
            "entry_logits": torch.zeros(1, 3),
            "side_logits": torch.tensor([[0.4, -0.2]]),
            "evasion_logits": torch.zeros(1, 2),
        }
        sample = {
            "ID": "sample",
            "entry_index": None,
            "side": 1,
            "evasion": None,
        }
        task_weights = {"entry": 1.0, "side": 1.0, "evasion": 1.0}
        unweighted, _ = _combined_loss(
            outputs,
            sample,
            nn.Identity(),
            nn.CrossEntropyLoss(reduction="none"),
            nn.CrossEntropyLoss(reduction="none"),
            task_weights,
            torch.device("cpu"),
        )
        weighted, _ = _combined_loss(
            outputs,
            sample,
            nn.Identity(),
            nn.CrossEntropyLoss(
                weight=torch.tensor([1.0, 3.0]), reduction="none"
            ),
            nn.CrossEntropyLoss(reduction="none"),
            task_weights,
            torch.device("cpu"),
        )
        torch.testing.assert_close(weighted, unweighted * 3)

    def test_zero_weight_tasks_are_excluded_from_total(self):
        outputs = {
            "entry_logits": torch.zeros(1, 3),
            "side_logits": torch.zeros(1, 2),
            "evasion_logits": torch.tensor([[0.2, -0.3]]),
        }
        sample = {
            "ID": "sample",
            "entry_index": None,
            "side": 1,
            "evasion": 0,
        }
        total, terms = _combined_loss(
            outputs,
            sample,
            nn.Identity(),
            nn.CrossEntropyLoss(reduction="none"),
            nn.CrossEntropyLoss(reduction="none"),
            {"entry": 0.0, "side": 0.0, "evasion": 1.0},
            torch.device("cpu"),
        )
        expected = nn.CrossEntropyLoss()(outputs["evasion_logits"], torch.tensor([0]))
        torch.testing.assert_close(total, expected)
        self.assertIn("side", terms)
        self.assertIn("evasion", terms)


if __name__ == "__main__":
    unittest.main()
