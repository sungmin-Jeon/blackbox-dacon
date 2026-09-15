"""CPU checks for clip-consistent Stage 1 training augmentation."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

from inference import S1_MEAN, S1_STD
from src.stage1.augmentation import augmentation_config, augment_stage1_clip
from src.stage1.data.build import build_direct_datasets


class AugmentationTests(unittest.TestCase):
    @staticmethod
    def _normalized_clip(frames: int = 4) -> torch.Tensor:
        horizontal = torch.linspace(0.2, 0.8, 8)[None, None, None, :]
        pixels = horizontal.expand(3, frames, 6, 8).clone()
        return (pixels - S1_MEAN) / S1_STD

    def test_none_is_exactly_unchanged(self):
        clip = self._normalized_clip()
        self.assertIs(augment_stage1_clip(clip), clip)

    def test_weak_transform_is_seeded_and_clip_consistent(self):
        clip = self._normalized_clip()
        settings = augmentation_config(mode="weak")

        torch.manual_seed(42)
        first = augment_stage1_clip(clip, settings)
        torch.manual_seed(42)
        second = augment_stage1_clip(clip, settings)
        torch.testing.assert_close(first, second)
        for frame_index in range(1, clip.shape[1]):
            torch.testing.assert_close(first[:, frame_index], first[:, 0])

        torch.manual_seed(7)
        third = augment_stage1_clip(clip, settings)
        self.assertFalse(torch.equal(first, third))

    def test_horizontal_flip_can_be_isolated(self):
        clip = self._normalized_clip()
        settings = augmentation_config(
            mode="weak",
            flip_probability=1.0,
            brightness=0.0,
            contrast=0.0,
            saturation=0.0,
        )
        torch.testing.assert_close(
            augment_stage1_clip(clip, settings),
            torch.flip(clip, dims=(-1,)),
        )

    def test_bad_settings_are_rejected(self):
        for kwargs in (
            {"mode": "strong"},
            {"flip_probability": -0.1},
            {"flip_probability": 1.1},
            {"brightness": -0.1},
            {"contrast": 1.1},
            {"saturation": -0.1},
            {"version": 999},
        ):
            with self.assertRaises(ValueError):
                augmentation_config(**kwargs)

    def test_dataset_augments_train_only_without_disabling_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "placeholder.mp4"
            video.touch()
            manifest = root / "split.csv"
            rows = [
                {
                    "source_id": f"R{index:03d}",
                    "video_id": f"{split}_{label}",
                    "kind": "source" if label == "ORIGINAL" else "recaptured",
                    "label": label,
                    "split": split,
                    "path": str(video),
                }
                for index, split in enumerate(("train", "val"), 1)
                for label in ("ORIGINAL", "RERECORDED")
            ]
            pd.DataFrame(rows).to_csv(manifest, index=False)
            train, val = build_direct_datasets(
                manifest,
                frames=4,
                size=8,
                cache_dir=root / "cache",
                augmentation=augmentation_config(
                    mode="weak",
                    flip_probability=1.0,
                    brightness=0.0,
                    contrast=0.0,
                    saturation=0.0,
                ),
            )
            base = self._normalized_clip()

            self.assertIsNotNone(train._cache_path(train.samples[0]))
            with patch.object(train, "_load_clip", return_value=base):
                augmented, _ = train[0]
            with patch.object(val, "_load_clip", return_value=base):
                evaluated, _ = val[0]
            torch.testing.assert_close(augmented, torch.flip(base, dims=(-1,)))
            self.assertIs(evaluated, base)


if __name__ == "__main__":
    unittest.main()
