"""Numerical and gradient checks for the selectable Stage 2 objectives."""

import unittest

import torch
from torch.nn import functional as F

from src.stage2.losses import (
    TEMPORAL_LOSS_NAMES,
    make_classification_loss,
    make_regression_loss,
    make_temporal_loss,
)


def sample(length=11, target=5):
    return {
        "frames": torch.arange(1, length + 1),
        "target_index": target,
        "target_frame": target + 1,
        "fps": 15.0,
        "target_min_frame": max(1, target),
        "target_max_frame": min(length, target + 2),
    }


class TemporalLossTests(unittest.TestCase):
    def test_hard_ce_matches_pytorch_value_and_gradient(self):
        scores = torch.tensor([0.3, -0.2, 1.2], dtype=torch.double, requires_grad=True)
        actual = make_temporal_loss()(scores, sample(3, 1))
        expected = F.cross_entropy(scores[None], torch.tensor([1]))
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual, scores)[0],
            torch.autograd.grad(expected, scores)[0],
        )
        torch.testing.assert_close(make_temporal_loss()(scores[None], sample(3, 1)), expected)

    def test_all_losses_have_finite_gradients_at_boundaries_and_single_frame(self):
        for name in TEMPORAL_LOSS_NAMES:
            for length, target in [(1, 0), (11, 0), (11, 5), (11, 10)]:
                with self.subTest(name=name, length=length, target=target):
                    scores = torch.linspace(-4, 4, length, requires_grad=True)
                    loss = make_temporal_loss(name)(scores, sample(length, target))
                    self.assertEqual(loss.shape, torch.Size([]))
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    self.assertTrue(torch.isfinite(scores.grad).all())

    def test_point_targets_reduce_to_hard_ce(self):
        scores = torch.tensor([0.1, 0.8, -0.2])
        s = sample(3, 1)
        s.update(target_min_frame=2, target_max_frame=2)
        expected = make_temporal_loss()(scores, s)
        for name, kwargs in [
            ("window_ce", {"radius_sec": 0}), ("window_nll", {"radius_sec": 0}),
            ("annotation_ce", {}), ("annotation_nll", {}),
            ("focal_ce", {"gamma": 0}), ("label_smoothing", {"smoothing": 0}),
        ]:
            with self.subTest(name=name):
                torch.testing.assert_close(make_temporal_loss(name, **kwargs)(scores, s), expected)

    def test_window_uses_seconds_and_actual_frame_numbers(self):
        scores = torch.arange(11, dtype=torch.double)
        logp = scores.log_softmax(0)
        # At 15 FPS, +/-4 frames fit; +/-5 frames exceed 0.3 seconds.
        torch.testing.assert_close(
            make_temporal_loss("window_ce")(scores, sample()), -logp[1:10].mean()
        )
        # Non-consecutive filenames: array adjacency is not elapsed time.
        s = {"frames": [101, 103, 107], "target_index": 1, "target_frame": 103, "fps": 10}
        torch.testing.assert_close(
            make_temporal_loss("window_ce")(scores[:3], s), -scores[:3].log_softmax(0)[:2].mean()
        )

    def test_timestamps_override_fps_and_include_exact_boundary(self):
        scores = torch.tensor([0.1, 0.7, -0.3, 1.2], dtype=torch.double)
        s = {"frames": [1, 2, 3, 4], "target_index": 1,
             "frame_times": [5.0, 5.1, 5.4, 5.4001], "fps": 0}
        torch.testing.assert_close(
            make_temporal_loss("window_ce")(scores, s), -scores.log_softmax(0)[:3].mean()
        )

    def test_gaussian_kl_and_ce_have_same_gradients(self):
        scores = torch.linspace(-1, 1, 11, dtype=torch.double, requires_grad=True)
        ce = make_temporal_loss("gaussian_ce")(scores, sample())
        kl = make_temporal_loss("gaussian_kl")(scores, sample())
        q = torch.softmax(-0.5 * ((torch.arange(11, dtype=torch.double) - 5) / 15 / 0.1)**2, 0)
        torch.testing.assert_close(ce - kl, -(q * q.log()).sum())
        torch.testing.assert_close(torch.autograd.grad(ce, scores)[0], torch.autograd.grad(kl, scores)[0])

    def test_interval_mass_is_different_from_uniform_target(self):
        scores = torch.tensor([-20., 20., -20., -20., -20.])
        s = sample(5, 2)
        self.assertLess(make_temporal_loss("window_nll", radius_sec=0.1)(scores, s).item(), 1e-5)
        self.assertGreater(make_temporal_loss("window_ce", radius_sec=0.1)(scores, s).item(), 20)

    def test_wasserstein_equals_integrated_absolute_cdf_error(self):
        scores = torch.tensor([0.2, 0.8, -0.3, 1.1], dtype=torch.double)
        times = torch.tensor([0., 0.1, 0.4, 1.2], dtype=torch.double)
        s = sample(4, 1)
        s["frame_times"] = times
        target_cdf = torch.tensor([0., 1., 1., 1.], dtype=torch.double)
        expected = ((scores.softmax(0).cumsum(0) - target_cdf).abs()[:-1] * times.diff()).sum()
        torch.testing.assert_close(make_temporal_loss("wasserstein1")(scores, s), expected)

    def test_focal_bce_gamma_zero_matches_bce_and_balanced_weight(self):
        scores = torch.linspace(-1, 1, 11)
        torch.testing.assert_close(
            make_temporal_loss("focal_bce", gamma=0)(scores, sample()),
            make_temporal_loss("bce")(scores, sample()),
        )
        target = F.one_hot(torch.tensor(5), num_classes=11).float()
        torch.testing.assert_close(
            make_temporal_loss("bce", pos_weight="balanced")(scores, sample()),
            F.binary_cross_entropy_with_logits(scores, target, pos_weight=torch.tensor(10.)),
        )

    def test_ordinal_matches_cumulative_binary_cross_entropy(self):
        scores = torch.tensor([0.2, 0.8, -0.3, 1.1], dtype=torch.double)
        expected = F.binary_cross_entropy(scores.softmax(0).cumsum(0)[:-1], torch.tensor([0., 1., 1.], dtype=torch.double))
        torch.testing.assert_close(make_temporal_loss("ordinal_bce")(scores, sample(4, 1)), expected)
        extreme = torch.tensor([-1000., 1000., -1000.], requires_grad=True)
        loss = make_temporal_loss("ordinal_bce")(extreme, sample(3, 2))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(extreme.grad).all())

    def test_invalid_labels_and_configs_fail_explicitly(self):
        scores = torch.zeros(3)
        changes = [
            {"target_index": None}, {"target_index": float("nan")}, {"target_index": 1.5},
            {"target_index": -1}, {"target_index": 3}, {"fps": 0}, {"fps": float("nan")},
            {"frames": [1, 1, 3]}, {"frames": [1, 2]}, {"target_frame": 3},
            {"frame_times": [0, 0.1, 0.05]},
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                make_temporal_loss()(scores, {**sample(3, 1), **change})
        with self.assertRaises(ValueError):
            make_temporal_loss("not_a_loss")
        with self.assertRaises(TypeError):
            make_temporal_loss("gaussian_ce", sigma=0.1)
        with self.assertRaises(ValueError):
            make_temporal_loss("gaussian_ce", sigma_sec=0)(scores, sample(3, 1))
        with self.assertRaises(ValueError):
            make_temporal_loss("annotation_ce")(scores, {**sample(3, 1), "target_min_frame": None})
        with self.assertRaises(ValueError):
            make_temporal_loss()(torch.zeros(2, 3), sample(3, 1))

    def test_config_round_trip(self):
        original = make_temporal_loss("focal_bce", gamma=1.5)
        self.assertEqual(original.config, {"name": "focal_bce", "gamma": 1.5, "alpha": None, "radius_sec": 0.0})
        restored = make_temporal_loss(**original.config)
        scores = torch.linspace(-1, 1, 11)
        torch.testing.assert_close(original(scores, sample()), restored(scores, sample()))


class OtherLossTests(unittest.TestCase):
    def test_classification_focal_gamma_zero_and_macro_f1_limit(self):
        scores = torch.tensor([[20., -20.], [20., -20.], [-20., 20.], [-20., 20.]], requires_grad=True)
        target = torch.tensor([0, 1, 1, 1])
        torch.testing.assert_close(make_classification_loss("focal", gamma=0)(scores, target), F.cross_entropy(scores, target))
        # Class F1 values are 2/3 and 4/5, including both defined classes.
        loss = make_classification_loss("soft_macro_f1")(scores, target)
        torch.testing.assert_close(loss, torch.tensor(1 - (2/3 + 4/5) / 2))
        loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_regression_losses_backpropagate_in_time_units(self):
        for name in ["l1", "mse", "huber", "smooth_l1"]:
            with self.subTest(name=name):
                predicted = torch.tensor([0.2, 0.7], requires_grad=True)
                loss = make_regression_loss(name)(predicted, torch.tensor([0.4, 0.6]))
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(torch.isfinite(predicted.grad).all())
                self.assertLess(predicted.grad[0], 0)
                self.assertGreater(predicted.grad[1], 0)


if __name__ == "__main__":
    unittest.main()
