"""CPU checks for clip selection, caching and checkpoint/evaluation wiring."""

import tempfile
import unittest
import contextlib
import io
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
import torch

from inference import (
    S1_MEAN, S1_STD, _crop_candidates, _decode_stage1_clip,
    _fft_patch_score, _spatial_config, _stage1_spatial_config, _Stage1Clips,
    _spatial_cache_tag, _stage1_temporal_config, _temporal_config,
)
from eval_stage1 import Stage1VideoDataset
from src.stage1.data.build import build_direct_dataloaders
from src.stage1.experiment import parse_args, run


class SpatialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.rng = np.random.default_rng(9)
        self.rgb = self.rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)

    def tearDown(self):
        self.temp.cleanup()

    def decode(self, mode, training=False, rgb=None, name="R001.mp4"):
        rgb = self.rgb if rgb is None else rgb
        with patch("inference._read_stage1_frames", side_effect=lambda *_: iter([rgb] * 16)):
            return _decode_stage1_clip(
                Path(name), 32, range(16), training=training,
                spatial=_spatial_config(mode=mode, crop_size=32, grid_size=3),
            )

    def test_legacy_is_unchanged(self):
        resized = cv2.resize(self.rgb, (48, 32), interpolation=cv2.INTER_AREA)
        expected = torch.from_numpy(resized[:, 8:40].copy()).permute(2, 0, 1).float() / 255
        expected = (expected[:, None] - S1_MEAN) / S1_STD
        clip = self.decode("center")
        self.assertTrue(torch.equal(clip, expected.expand(-1, 16, -1, -1)))
        self.assertEqual(_stage1_spatial_config({})["mode"], "center")

    def test_shape_temporal_consistency_and_reproducible_evaluation(self):
        for mode in ("native-center", "random", "fft"):
            with self.subTest(mode=mode):
                clip = self.decode(mode)
                self.assertEqual(tuple(clip.shape), (3, 16, 32, 32))
                self.assertTrue(torch.equal(clip, clip[:, :1].expand_as(clip)))
                self.assertTrue(torch.equal(clip, self.decode(mode, name="/different/root/R001.mp4")))
        np.random.seed(42)
        a = self.decode("random", training=True)
        np.random.seed(42)
        self.assertTrue(torch.equal(a, self.decode("random", training=True)))
        draws = [self.decode("random", training=True) for _ in range(10)]
        self.assertTrue(any(not torch.equal(a, b) for b in draws))

    def test_fft_score_and_clip_aggregation(self):
        stripes = np.tile((np.arange(32) % 2 * 255).astype(np.uint8), (32, 1))
        stripes = np.repeat(stripes[:, :, None], 3, axis=2)
        blank = np.full_like(stripes, 128)
        self.assertGreater(_fft_patch_score(stripes), _fft_patch_score(blank))
        first = np.concatenate([stripes, blank], axis=1)
        later = np.concatenate([blank, stripes], axis=1)
        frames = [first] + [later] * 15
        with patch("inference._read_stage1_frames", side_effect=lambda *_: iter(frames)):
            clip = _decode_stage1_clip(Path("x.mp4"), 32, range(16),
                                       spatial=_spatial_config(mode="fft", crop_size=32, grid_size=2))
        # The right crop wins over the clip despite the left winning frame 0.
        recovered = ((clip * S1_STD + S1_MEAN) * 255).round().byte()
        self.assertTrue(torch.equal(recovered[:, 0], torch.from_numpy(blank).permute(2, 0, 1)))
        self.assertTrue(torch.equal(recovered[:, 1], torch.from_numpy(stripes).permute(2, 0, 1)))

    def test_bad_settings_and_small_frames(self):
        for kwargs in ({"mode": "typo"}, {"version": 999}, {"grid_size": 0},
                       {"fft_min_freq": 0}, {"fft_min_freq": 0.5}):
            with self.assertRaises(ValueError):
                _spatial_config(**kwargs)
        with self.assertRaises(ValueError):
            _crop_candidates(10, 20, 32, 5)
        self.assertEqual(_crop_candidates(32, 32, 32, 5), [(0, 0)])

    def test_cache_key_tracks_selection_parameters(self):
        base = _spatial_config(mode="fft")
        configs = [base] + [{**base, **change} for change in (
            {"crop_size": 256}, {"grid_size": 3}, {"fft_min_freq": 0.3},
            {"seed": 7}, {"mode": "random"},
        )]
        self.assertEqual(len({_spatial_cache_tag(c) for c in configs}), len(configs))

    def test_training_entrypoint_saves_spatial_checkpoint(self):
        _, manifest = self.make_manifest()
        argv = ["train_stage1.py", "--dataset", "direct", "--split-csv", str(manifest),
                "--model-dir", str(self.root / "run"), "--epochs", "1",
                "--num-workers", "0", "--size", "32", "--spatial-mode", "fft",
                "--crop-size", "32", "--crop-grid", "3", "--no-pretrained", "--no-amp"]
        with patch("sys.argv", argv):
            args = parse_args()
        # Exercise optimizer, train/val loop and real checkpoint writer without
        # spending a full MViT training run on every unit-test invocation.
        model = torch.nn.Module()
        model.net = torch.nn.Sequential(torch.nn.AdaptiveAvgPool3d(1),
                                         torch.nn.Flatten(), torch.nn.Linear(3, 2))
        model.forward = model.net.forward
        with patch("src.stage1.experiment._build_model", return_value=(model, None)), \
             contextlib.redirect_stdout(io.StringIO()):
            path = run(args)
        checkpoint = torch.load(path, weights_only=False)
        self.assertEqual(_stage1_spatial_config(checkpoint),
                         _spatial_config(mode="fft", crop_size=32, grid_size=3))
        self.assertEqual(_stage1_temporal_config(checkpoint), _temporal_config())
        last = torch.load(path.parent / "last.pt", weights_only=False)
        self.assertEqual(_stage1_spatial_config(last), _stage1_spatial_config(checkpoint))
        self.assertIn("optimizer", last)

    def make_manifest(self):
        video = self.root / "R001.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 15, (96, 64))
        self.assertTrue(writer.isOpened())
        for _ in range(16):
            writer.write(cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR))
        writer.release()
        rows = [dict(source_id=f"R{i:03d}", video_id=f"{split}_{label}",
                     kind="source" if label == "ORIGINAL" else "recaptured",
                     label=label, split=split, path=str(video))
                for i, split in enumerate(("train", "val"), 1)
                for label in ("ORIGINAL", "RERECORDED")]
        manifest = self.root / "split.csv"
        pd.DataFrame(rows).to_csv(manifest, index=False)
        return video, manifest

    def test_real_decode_loaders_cache_and_inference_agree(self):
        video, manifest = self.make_manifest()
        paths = []
        for mode in ("center", "native-center", "random", "fft"):
            spatial = _spatial_config(mode=mode, crop_size=32, grid_size=3)
            train, val = build_direct_dataloaders(
                manifest, spatial=spatial, frames=16, size=32, num_workers=0,
                cache_dir=self.root / "cache",
            )
            batch, labels = next(iter(train))
            self.assertEqual(tuple(batch.shape), (2, 3, 16, 32, 32))
            self.assertEqual(tuple(labels.shape), (2,))
            sample = train.dataset.samples[0]
            cache_path = train.dataset._cache_path(sample)
            if mode == "random":
                self.assertIsNone(cache_path)
            else:
                train.dataset[0]
                self.assertTrue(cache_path.is_file())
                paths.append(cache_path)
            # Config survives checkpoints and is used consistently by both eval paths.
            checkpoint = self.root / "test.pt"
            torch.save({"config": {"spatial": spatial}}, checkpoint)
            restored = _stage1_spatial_config(torch.load(checkpoint, weights_only=True))
            evaluator = Stage1VideoDataset([{"path": video}], frames=16, size=32, spatial=restored)
            submission = _Stage1Clips([video], 1, 32, 16, spatial=restored)
            evaluated = evaluator[0][0]
            submitted, _, ok = submission[0]
            self.assertEqual(ok, 1)
            self.assertTrue(torch.equal(evaluated, submitted))
            torch.testing.assert_close(val.dataset[0][0], evaluated, atol=1e-3, rtol=1e-3)
            # Second read uses the fp16 deterministic cache.
            torch.testing.assert_close(val.dataset[0][0], evaluated, atol=1e-3, rtol=1e-3)
        self.assertEqual(len(set(paths)), 3)


if __name__ == "__main__":
    unittest.main()
