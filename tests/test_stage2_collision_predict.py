"""Checks collision-context prediction for directly labelled videos."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models import ResNet18_Weights, resnet18

from inference import (
    _Stage2CollisionBiGRU,
    _stage2_collision_model_from_checkpoint,
)
from src.stage2.collision_predict import (
    BaselineCollisionBiGRU,
    CollisionBiGRU,
    collision_model_from_checkpoint,
    predict_video,
)


class CollisionContextPredictionTests(unittest.TestCase):
    def test_model_matches_submission_collision_model(self):
        torch.manual_seed(3)
        submitted = _Stage2CollisionBiGRU(
            input_size=8, hidden_size=4, num_layers=1, dropout=0.0
        ).eval()
        context = CollisionBiGRU(
            input_size=8, hidden_size=4, num_layers=1, dropout=0.0
        ).eval()
        context.load_state_dict(submitted.state_dict())
        features = torch.randn(1, 6, 8)
        torch.testing.assert_close(context(features), submitted(features))

    def test_both_checkpoint_formats_are_detected(self):
        features = torch.randn(1, 6, 8)
        cases = [
            (
                CollisionBiGRU(8, 4, 1, 0.0).eval(),
                "model_state_dict",
                "collision_v0",
            ),
            (
                BaselineCollisionBiGRU(8, 4, 2).eval(),
                "model",
                "baseline",
            ),
        ]
        for original, checkpoint_key, expected_format in cases:
            with self.subTest(expected_format=expected_format):
                checkpoint = {checkpoint_key: original.state_dict()}
                restored, actual_format = collision_model_from_checkpoint(checkpoint)
                submitted, submitted_format = _stage2_collision_model_from_checkpoint(
                    checkpoint
                )
                self.assertEqual(actual_format, expected_format)
                self.assertEqual(submitted_format, expected_format)
                restored.eval()
                submitted.eval()
                torch.testing.assert_close(restored(features), original(features))
                torch.testing.assert_close(submitted(features), original(features))

    def test_video_prediction_returns_one_based_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.avi"
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"MJPG"), 15, (64, 48)
            )
            self.assertTrue(writer.isOpened())
            for frame_index in range(3):
                frame = np.full((48, 64, 3), frame_index * 50, dtype=np.uint8)
                writer.write(frame)
            writer.release()

            backbone = resnet18(weights=None)
            backbone.fc = torch.nn.Identity()
            temporal = CollisionBiGRU(
                input_size=512, hidden_size=4, num_layers=1, dropout=0.0
            ).eval()
            prediction, count, fps, confidence = predict_video(
                path,
                backbone.eval(),
                temporal,
                ResNet18_Weights.IMAGENET1K_V1.transforms(),
                torch.device("cpu"),
                batch_size=2,
            )
            self.assertIn(prediction, (1, 2, 3))
            self.assertEqual(count, 3)
            self.assertAlmostEqual(fps, 15.0)
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
