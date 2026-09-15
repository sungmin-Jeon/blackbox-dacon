"""CPU checks for Stage 1 temporal sampling and multi-view wiring."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from eval_stage1 import Stage1VideoDataset
from inference import (
    _Stage1Clips,
    _stage1_temporal_config,
    _temporal_config,
    _temporal_eval_views,
    _temporal_frame_ids,
)
from src.stage1.data.build import build_direct_datasets
from src.stage1.engine import validate


class TemporalTests(unittest.TestCase):
    def test_legacy_checkpoint_is_one_uniform_view(self):
        temporal = _stage1_temporal_config({})
        self.assertEqual(
            temporal,
            {
                "mode": "uniform",
                "eval_mode": "uniform",
                "bursts": 4,
                "target_fps": 15.0,
                "version": 1,
            },
        )
        self.assertEqual(_temporal_eval_views(temporal), ("uniform",))

    def test_auto_eval_mode_follows_training_mode(self):
        self.assertEqual(_temporal_config(mode="uniform")["eval_mode"], "uniform")
        self.assertEqual(
            _temporal_config(mode="multi-burst")["eval_mode"], "multi-burst"
        )
        mixed = _temporal_config(mode="mixed")
        self.assertEqual(mixed["eval_mode"], "both")
        self.assertEqual(_temporal_eval_views(mixed), ("uniform", "multi-burst"))

    def test_uniform_and_centered_multi_burst_indices(self):
        np.testing.assert_array_equal(
            _temporal_frame_ids(300, 16, mode="uniform"),
            np.linspace(0, 299, 16).round().astype(int),
        )
        expected = np.concatenate(
            [np.arange(start, start + 4) for start in (35, 110, 185, 260)]
        )
        actual = _temporal_frame_ids(300, 16, mode="multi-burst", bursts=4)
        np.testing.assert_array_equal(actual, expected)
        for burst in actual.reshape(4, 4):
            np.testing.assert_array_equal(np.diff(burst), np.ones(3, dtype=int))

    def test_v2_multi_burst_is_fixed_and_fps_normalized(self):
        configurations = ((150, 15, 1), (300, 30, 2), (600, 60, 4))
        for total, fps, expected_stride in configurations:
            with self.subTest(fps=fps):
                evaluated = _temporal_frame_ids(
                    total, 16, mode="multi-burst", bursts=4, fps=fps,
                )
                trained = _temporal_frame_ids(
                    total, 16, mode="multi-burst", bursts=4, fps=fps,
                    training=True,
                )
                np.testing.assert_array_equal(trained, evaluated)
                for burst in evaluated.reshape(4, 4):
                    np.testing.assert_array_equal(
                        np.diff(burst),
                        np.full(3, expected_stride, dtype=int),
                    )

    def test_v1_training_jitter_remains_checkpoint_compatible(self):
        np.random.seed(42)
        first = _temporal_frame_ids(
            300, 16, mode="multi-burst", bursts=4, training=True, version=1,
        )
        np.random.seed(42)
        np.testing.assert_array_equal(
            first,
            _temporal_frame_ids(
                300, 16, mode="multi-burst", bursts=4, training=True,
                version=1,
            ),
        )
        draws = [
            _temporal_frame_ids(
                300, 16, mode="multi-burst", bursts=4, training=True,
                version=1,
            )
            for _ in range(8)
        ]
        self.assertTrue(any(not np.array_equal(first, draw) for draw in draws))
        for burst_index, burst in enumerate(first.reshape(4, 4)):
            self.assertGreaterEqual(int(burst[0]), burst_index * 75)
            self.assertLess(int(burst[-1]), (burst_index + 1) * 75)
            np.testing.assert_array_equal(np.diff(burst), np.ones(3, dtype=int))

    def test_short_videos_and_bad_settings(self):
        frame_ids = _temporal_frame_ids(3, 16, mode="multi-burst", bursts=4)
        self.assertEqual(len(frame_ids), 16)
        self.assertTrue(np.all((0 <= frame_ids) & (frame_ids < 3)))
        self.assertTrue(np.all(frame_ids[:-1] <= frame_ids[1:]))

        for kwargs in (
            {"mode": "typo"},
            {"eval_mode": "typo"},
            {"bursts": 0},
            {"target_fps": 0},
            {"version": 999},
        ):
            with self.assertRaises(ValueError):
                _temporal_config(**kwargs)
        with self.assertRaises(ValueError):
            _temporal_frame_ids(300, 15, mode="multi-burst", bursts=4)
        with self.assertRaises(ValueError):
            _temporal_frame_ids(300, 16, mode="mixed", training=False)

    def test_direct_validation_and_submission_use_the_same_two_views(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "R001.avi"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"MJPG"), 15, (64, 48)
            )
            self.assertTrue(writer.isOpened())
            for frame_index in range(64):
                rgb = np.full((48, 64, 3), frame_index * 3, dtype=np.uint8)
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            writer.release()

            rows = [
                dict(
                    source_id=f"R{index:03d}",
                    video_id=f"{split}_{label}",
                    kind="source" if label == "ORIGINAL" else "recaptured",
                    label=label,
                    split=split,
                    path=str(video),
                )
                for index, split in enumerate(("train", "val"), 1)
                for label in ("ORIGINAL", "RERECORDED")
            ]
            manifest = root / "split.csv"
            pd.DataFrame(rows).to_csv(manifest, index=False)
            temporal = _temporal_config(mode="mixed", eval_mode="both", bursts=4)

            train, val = build_direct_datasets(
                manifest, frames=16, size=32, temporal=temporal
            )
            self.assertIsNone(train._cache_path(train.samples[0]))
            self.assertEqual(tuple(train[0][0].shape), (3, 16, 32, 32))
            self.assertEqual(tuple(val[0][0].shape), (2, 3, 16, 32, 32))

            samples = [{"path": video}]
            evaluator = Stage1VideoDataset(
                samples, frames=16, size=32, temporal=temporal
            )
            submission = _Stage1Clips(
                [video], 2, 32, 16, temporal=temporal
            )
            self.assertEqual(len(evaluator), 2)
            self.assertEqual(len(submission), 2)
            for index, expected_view in enumerate(("uniform", "multi-burst")):
                evaluated, video_index, view = evaluator[index]
                submitted, submitted_index, valid = submission[index]
                self.assertEqual((video_index, submitted_index, valid), (0, 0, 1))
                self.assertEqual(view, expected_view)
                self.assertTrue(torch.equal(evaluated, submitted))

    def test_fixed_v2_multi_burst_training_is_cacheable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "R001.avi"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48)
            )
            self.assertTrue(writer.isOpened())
            for frame_index in range(64):
                rgb = np.full((48, 64, 3), frame_index * 3, dtype=np.uint8)
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            writer.release()

            rows = [
                dict(
                    source_id=f"R{index:03d}",
                    video_id=f"{split}_{label}",
                    kind="source" if label == "ORIGINAL" else "recaptured",
                    label=label,
                    split=split,
                    path=str(video),
                )
                for index, split in enumerate(("train", "val"), 1)
                for label in ("ORIGINAL", "RERECORDED")
            ]
            manifest = root / "split.csv"
            pd.DataFrame(rows).to_csv(manifest, index=False)
            temporal = _temporal_config(mode="multi-burst", bursts=4)
            train, _ = build_direct_datasets(
                manifest,
                frames=16,
                size=32,
                cache_dir=root / "cache",
                temporal=temporal,
            )

            cache_path = train._cache_path(train.samples[0])
            self.assertIsNotNone(cache_path)
            first = train[0][0]
            self.assertTrue(cache_path.is_file())
            torch.testing.assert_close(train[0][0], first, atol=1e-3, rtol=1e-3)

    def test_validation_averages_view_probabilities(self):
        class MeanLogitModel(torch.nn.Module):
            def forward(self, clips):
                signal = clips.mean(dim=(1, 2, 3, 4))
                return torch.stack((-signal, signal), dim=1)

        clips = torch.tensor(
            [
                [[[[[-2.0]]]], [[[[1.0]]]]],
                [[[[[2.0]]]], [[[[-1.0]]]]],
            ]
        )
        labels = torch.tensor([0, 1])
        loader = DataLoader(TensorDataset(clips, labels), batch_size=2)
        result = validate(
            MeanLogitModel(), loader, torch.nn.CrossEntropyLoss(),
            torch.device("cpu"), amp=False,
        )
        self.assertEqual(result.metrics.accuracy, 1.0)

        logits = MeanLogitModel()(clips.reshape(4, 1, 1, 1, 1)).reshape(2, 2, 2)
        probabilities = torch.softmax(logits, dim=2).mean(dim=1)
        expected_loss = torch.nn.functional.nll_loss(probabilities.log(), labels)
        self.assertAlmostEqual(result.loss, expected_loss.item(), places=6)


if __name__ == "__main__":
    unittest.main()
