"""Checks collision models used by the feature comparison experiment."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from src.stage2.collision_feature_compare import (
    CollisionFeatureModel,
    load_feature_samples,
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


if __name__ == "__main__":
    unittest.main()
