"""Checks collision models used by the feature comparison experiment."""

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import pandas as pd
import torch

from src.stage2.collision_feature_compare import (
    CollisionFeatureModel,
    load_feature_samples,
    train_one_feature,
)


class CollisionFeatureCompareTests(unittest.TestCase):
    def test_global_and_spatial_models_return_frame_logits(self):
        cases = [
            ("global", 512, torch.randn(2, 5, 512)),
            ("spatial", 256, torch.randn(2, 5, 256, 4, 6)),
            ("spatial", 512, torch.randn(2, 5, 512, 2, 3)),
        ]
        for feature_kind, channels, features in cases:
            with self.subTest(feature_kind=feature_kind, channels=channels):
                model = CollisionFeatureModel(
                    feature_kind=feature_kind,
                    input_channels=channels,
                    projection_size=16,
                    temporal_input_size=24,
                    hidden_size=8,
                    num_layers=1,
                    dropout=0.0,
                )
                self.assertEqual(model(features).shape, (2, 5))

    def test_loader_detects_feature_kind_and_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = pd.DataFrame(
                [
                    {
                        "ID": "sample",
                        "split": "train",
                        "collision_frame": 3,
                    }
                ]
            )
            for name, features, expected_kind in (
                ("global", torch.randn(4, 512), "global"),
                ("spatial", torch.randn(4, 256, 3, 5), "spatial"),
            ):
                feature_dir = root / name
                feature_dir.mkdir()
                torch.save(
                    {
                        "features": features,
                        "frame_numbers": torch.arange(1, 5),
                        "fps": 15.0,
                    },
                    feature_dir / "sample.pt",
                )
                samples, info = load_feature_samples(labels, feature_dir)
                self.assertEqual(info["feature_kind"], expected_kind)
                self.assertEqual(samples[0]["target_index"], 2)

    def test_training_saves_accuracy_and_loss_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_dir = root / "global_center"
            feature_dir.mkdir()
            labels = pd.DataFrame(
                [
                    {
                        "ID": f"sample_{index}",
                        "split": "train" if index < 2 else "val",
                        "collision_frame": 3,
                    }
                    for index in range(3)
                ]
            )
            for video_id in labels["ID"]:
                torch.save(
                    {
                        "features": torch.randn(5, 8),
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
                sigma_sec=0.1,
                lr=2e-4,
                weight_decay=1e-4,
                epochs=1,
                early_patience=2,
                early_min_delta=1e-4,
                grad_clip=1.0,
            )
            result = train_one_feature(
                feature_name="global_center",
                feature_dir=feature_dir,
                labels=labels,
                run_dir=run_dir,
                args=args,
                device=torch.device("cpu"),
            )
            output = run_dir / "global_center"
            self.assertTrue((output / "best_model.pt").is_file())
            self.assertTrue((output / "best_accuracy_model.pt").is_file())
            self.assertTrue((output / "best_loss_model.pt").is_file())
            self.assertEqual(result["best_epoch"], 1)
            self.assertEqual(result["best_loss_epoch"], 1)


if __name__ == "__main__":
    unittest.main()
