"""Checks that submission Stage 2 matches the trained Direct model."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader as TorchDataLoader
from torchvision.models import resnet18

import inference
from src.stage2.collision_feature_compare import CollisionFeatureModel
from src.stage2.direct_model import Stage2DirectSpatial
from src.stage2.entry_feature_compare import EntrySideFeatureModel
from src.stage2.spatial_extract import resize_short_edge


class Stage2InferenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.direct_config = {
            "input_channels": 8,
            "projection_size": 6,
            "temporal_input_size": 10,
            "hidden_size": 5,
            "num_layers": 1,
            "dropout": 0.0,
            "collision_window": 2,
            "entry_temperature": 1.0,
        }

    def test_direct_submission_model_matches_training_model(self):
        trained = Stage2DirectSpatial(**self.direct_config).eval()
        submitted = inference._Stage2DirectSpatial(**self.direct_config).eval()
        submitted.load_state_dict(trained.state_dict())
        maps = torch.randn(2, 7, 8, 4, 6)
        collision = torch.tensor([3, -1])
        expected = trained(maps, collision)
        entry, side, evasion = submitted(maps, collision)
        torch.testing.assert_close(entry, expected["entry_logits"])
        torch.testing.assert_close(side, expected["side_logits"])
        torch.testing.assert_close(evasion, expected["evasion_logits"])

    def test_resize_matches_feature_extraction(self):
        images = torch.randn(2, 3, 63, 101)
        torch.testing.assert_close(
            inference._stage2_resize_short_edge(images, 64),
            resize_short_edge(images, 64),
        )

    def test_final_collision_submission_model_matches_training_model(self):
        config = {
            "feature_kind": "spatial",
            "input_channels": 8,
            "projection_size": 4,
            "temporal_input_size": 6,
            "hidden_size": 3,
            "num_layers": 1,
            "dropout": 0.0,
            "delta_mode": "concat",
            "spatial_coordinates": False,
        }
        trained = CollisionFeatureModel(**config).eval()
        submitted = inference._Stage2FeatureTemporal(**config).eval()
        submitted.load_state_dict(trained.state_dict())
        maps = torch.randn(2, 7, 8, 3, 5)
        torch.testing.assert_close(submitted(maps), trained(maps))

    def test_final_entry_side_submission_model_matches_training_model(self):
        config = {
            "feature_kind": "spatial",
            "input_channels": 8,
            "projection_size": 4,
            "temporal_input_size": 6,
            "hidden_size": 3,
            "num_layers": 1,
            "dropout": 0.0,
            "delta_mode": "concat",
            "spatial_coordinates": False,
            "entry_temperature": 1.0,
            "side_context": "independent",
        }
        trained = EntrySideFeatureModel(**config).eval()
        submitted = inference._Stage2EntrySideTemporal(**config).eval()
        submitted.load_state_dict(trained.state_dict())
        maps = torch.randn(2, 7, 8, 3, 5)
        expected = trained(maps)
        entry, side = submitted(maps)
        torch.testing.assert_close(entry, expected["entry_logits"])
        torch.testing.assert_close(side, expected["side_logits"])

    def test_predict_stage2_uses_both_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "data" / "images" / "case_001"
            model_dir = root / "model"
            image_dir.mkdir(parents=True)
            model_dir.mkdir()

            for frame_number in (11, 12, 13):
                pixels = np.full((48, 72, 3), frame_number * 10, dtype=np.uint8)
                Image.fromarray(pixels).save(image_dir / f"{frame_number:06d}.jpg")

            backbone = resnet18(weights=None)
            torch.save(backbone.state_dict(), model_dir / "resnet18-f37072fd.pth")

            collision_config = {
                "input_size": 512,
                "hidden_size": 4,
                "num_layers": 1,
                "dropout": 0.0,
            }
            collision_model = inference._Stage2CollisionBiGRU(**collision_config)
            torch.save(
                {**collision_config, "model_state_dict": collision_model.state_dict()},
                model_dir / "best.pt",
            )

            direct_config = {
                **self.direct_config,
                "input_channels": 256,
            }
            direct_model = Stage2DirectSpatial(**direct_config)
            torch.save(
                {
                    "model_config": direct_config,
                    "model_state_dict": direct_model.state_dict(),
                    "feature_config": {
                        "layer": "layer3",
                        "short_edge": 64,
                        "crop": None,
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225],
                    },
                },
                model_dir / "direct.pt",
            )

            def loader_without_workers(*args, **kwargs):
                kwargs["num_workers"] = 0
                return TorchDataLoader(*args, **kwargs)

            with patch("inference._device", return_value=torch.device("cpu")), patch(
                "inference.DataLoader", side_effect=loader_without_workers
            ):
                result = inference.predict_stage2(root / "data", model_dir)

            self.assertEqual(result["ID"].tolist(), ["case_001"])
            self.assertIn(int(result.loc[0, "collision_frame"]), (11, 12, 13))
            self.assertIn(int(result.loc[0, "entry_frame"]), (11, 12, 13))
            self.assertIn(int(result.loc[0, "evasion_space"]), (0, 1))
            self.assertIn(result.loc[0, "entry_side"], ("LEFT", "RIGHT"))


if __name__ == "__main__":
    unittest.main()
