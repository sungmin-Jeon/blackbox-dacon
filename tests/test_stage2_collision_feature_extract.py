"""Checks the standalone Stage 2 collision feature manifest."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from src.stage2.collision_feature_extract import build_manifest


class CollisionFeatureExtractTests(unittest.TestCase):
    def test_manifest_validates_target_and_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save(
                {
                    "features": torch.randn(4, 512),
                    "frame_numbers": torch.arange(1, 5),
                    "fps": 15.0,
                },
                root / "video_a.pt",
            )
            labels = pd.DataFrame(
                [
                    {
                        "ID": "video_a",
                        "split": "val",
                        "collision_frame": 3,
                    }
                ]
            )
            manifest = build_manifest(labels, root)
            self.assertEqual(manifest.loc[0, "ID"], "video_a")
            self.assertEqual(manifest.loc[0, "target_index"], 2)
            self.assertEqual(manifest.loc[0, "feature_shape"], "4x512")
            self.assertEqual(manifest.loc[0, "first_frame"], 1)
            self.assertEqual(manifest.loc[0, "last_frame"], 4)


if __name__ == "__main__":
    unittest.main()
