"""Shape, gradient and boundary checks for the direct Stage 2 model."""

import unittest

import torch

from src.stage2.direct_model import Stage2DirectSpatial, direct_model_from_checkpoint


class DirectSpatialTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.config = {
            "input_channels": 8,
            "projection_size": 6,
            "temporal_input_size": 10,
            "hidden_size": 5,
            "num_layers": 1,
            "dropout": 0.0,
            "collision_window": 2,
            "entry_temperature": 1.0,
        }

    def test_outputs_and_attention_are_valid(self):
        model = Stage2DirectSpatial(**self.config)
        maps = torch.randn(2, 7, 8, 4, 6)
        outputs = model(maps, torch.tensor([0, 6]), return_attention=True)
        self.assertEqual(tuple(outputs["entry_logits"].shape), (2, 7))
        self.assertEqual(tuple(outputs["side_logits"].shape), (2, 2))
        self.assertEqual(tuple(outputs["evasion_logits"].shape), (2, 2))
        torch.testing.assert_close(
            outputs["spatial_attention"].sum(dim=(-2, -1)), torch.ones(2, 7)
        )
        torch.testing.assert_close(outputs["entry_attention"].sum(dim=1), torch.ones(2))

    def test_all_heads_backpropagate_into_spatial_pool(self):
        model = Stage2DirectSpatial(**self.config)
        maps = torch.randn(1, 7, 8, 4, 6, requires_grad=True)
        outputs = model(maps, torch.tensor([3]))
        loss = sum(value.square().mean() for value in outputs.values())
        loss.backward()
        self.assertTrue(torch.isfinite(maps.grad).all())
        for name in ("entry_head.weight", "side_head.3.weight", "evasion_head.3.weight"):
            self.assertIsNotNone(dict(model.named_parameters())[name].grad)

    def test_invalid_collision_indices_fail(self):
        model = Stage2DirectSpatial(**self.config)
        maps = torch.randn(1, 7, 8, 4, 6)
        for indices in (torch.tensor([-1]), torch.tensor([7]), torch.tensor([1, 2])):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                model(maps, indices)

    def test_checkpoint_round_trip(self):
        model = Stage2DirectSpatial(**self.config).eval()
        checkpoint = {"model_config": self.config, "model_state_dict": model.state_dict()}
        restored = direct_model_from_checkpoint(checkpoint).eval()
        maps = torch.randn(1, 7, 8, 3, 5)
        expected = model(maps, torch.tensor([2]))
        actual = restored(maps, torch.tensor([2]))
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])


if __name__ == "__main__":
    unittest.main()
