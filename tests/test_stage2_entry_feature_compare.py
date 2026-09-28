"""Checks Entry feature loading and paired temporal-prior metrics."""

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import pandas as pd
import torch

from src.stage2.entry_feature_compare import (
    EntrySideFeatureModel,
    entry_metrics,
    entry_model_from_checkpoint,
    load_feature_samples,
    train_one_feature,
)


class EntryFeatureCompareTests(unittest.TestCase):
    def test_entry_side_model_uses_entry_context_and_backpropagates(self):
        model = EntrySideFeatureModel(
            feature_kind="spatial",
            input_channels=8,
            projection_size=4,
            temporal_input_size=6,
            hidden_size=3,
            num_layers=1,
            dropout=0.0,
            delta_mode="concat",
        )
        inputs = torch.randn(2, 5, 8, 3, 4, requires_grad=True)
        outputs = model(inputs)
        self.assertEqual(outputs["entry_logits"].shape, (2, 5))
        self.assertEqual(outputs["side_logits"].shape, (2, 2))
        outputs["side_logits"].square().mean().backward()
        self.assertGreater(inputs.grad.abs().sum().item(), 0)
        self.assertGreater(model.collision_head.weight.grad.abs().sum().item(), 0)

    def test_entry_side_model_can_learn_independent_temporal_attention(self):
        model = EntrySideFeatureModel(
            feature_kind="spatial",
            input_channels=8,
            projection_size=4,
            temporal_input_size=6,
            hidden_size=3,
            num_layers=1,
            dropout=0.0,
            delta_mode="concat",
            side_context="independent",
        )
        inputs = torch.randn(2, 5, 8, 3, 4, requires_grad=True)
        outputs = model(inputs)
        self.assertEqual(outputs["side_attention"].shape, (2, 5))
        torch.testing.assert_close(
            outputs["side_attention"].sum(dim=1), torch.ones(2)
        )
        outputs["side_logits"].square().mean().backward()
        self.assertGreater(model.side_attention.weight.grad.abs().sum().item(), 0)
        self.assertGreater(inputs.grad.abs().sum().item(), 0)

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
                        "entry_side": "right",
                    }
                ]
            )
            samples, info = load_feature_samples(labels, root, {"sample": 2})
            self.assertEqual(info["feature_kind"], "spatial")
            self.assertEqual(samples[0]["target_index"], 2)
            self.assertEqual(samples[0]["prior_frame"], 2)
            self.assertEqual(samples[0]["entry_confidence"], "HIGH")
            self.assertEqual(samples[0]["side"], 1)

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

    def test_entry_training_saves_concat_delta_in_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_dir = root / "spatial_layer3"
            feature_dir.mkdir()
            labels = pd.DataFrame(
                [
                    {
                        "ID": f"sample_{index}",
                        "split": "train" if index < 2 else "val",
                        "entry_frame": 3,
                        "entry_confidence": "HIGH",
                        "entry_side": "LEFT" if index % 2 == 0 else "RIGHT",
                    }
                    for index in range(3)
                ]
            )
            for video_id in labels["ID"]:
                torch.save(
                    {
                        "features": torch.randn(5, 8, 3, 4),
                        "frame_numbers": torch.arange(1, 6),
                        "fps": 15.0,
                    },
                    feature_dir / f"{video_id}.pt",
                )
            run_dir = root / "run"
            run_dir.mkdir()
            args = Namespace(
                seed=42,
                no_amp=True,
                projection_size=4,
                temporal_input_size=6,
                hidden_size=3,
                num_layers=1,
                dropout=0.0,
                delta_mode="concat",
                spatial_coordinates=True,
                side_loss_weight=0.3,
                entry_temperature=1.0,
                side_context="independent",
                sigma_sec=0.1,
                lr=2e-4,
                weight_decay=1e-4,
                epochs=1,
                early_patience=2,
                early_min_delta=1e-4,
                grad_clip=1.0,
            )
            result = train_one_feature(
                feature_name="spatial_layer3",
                feature_dir=feature_dir,
                labels=labels,
                prior_by_id={"sample_2": 3},
                run_dir=run_dir,
                args=args,
                device=torch.device("cpu"),
            )
            checkpoint = torch.load(
                result["checkpoint"], map_location="cpu", weights_only=True
            )
            self.assertEqual(checkpoint["model_config"]["delta_mode"], "concat")
            self.assertTrue(checkpoint["model_config"]["spatial_coordinates"])
            self.assertEqual(checkpoint["model_class"], "EntrySideFeatureModel")
            self.assertEqual(checkpoint["model_config"]["side_context"], "independent")
            loaded = entry_model_from_checkpoint(checkpoint)
            self.assertIsInstance(loaded, EntrySideFeatureModel)
            self.assertIn("side_macro_f1", checkpoint["metrics"])


if __name__ == "__main__":
    unittest.main()
