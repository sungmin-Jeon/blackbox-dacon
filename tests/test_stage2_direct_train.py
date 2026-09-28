"""Checks for Stage 2 Direct training behavior."""

import unittest

import torch
from torch import nn

from src.stage2.direct_train import _combined_loss


class DirectTrainingTests(unittest.TestCase):
    def test_single_sample_class_weight_is_not_cancelled(self):
        outputs = {
            "entry_logits": torch.zeros(1, 3),
            "side_logits": torch.tensor([[0.4, -0.2]]),
            "evasion_logits": torch.zeros(1, 2),
        }
        sample = {
            "ID": "sample",
            "entry_index": None,
            "side": 1,
            "evasion": None,
        }
        task_weights = {"entry": 1.0, "side": 1.0, "evasion": 1.0}
        unweighted, _ = _combined_loss(
            outputs,
            sample,
            nn.Identity(),
            nn.CrossEntropyLoss(reduction="none"),
            nn.CrossEntropyLoss(reduction="none"),
            task_weights,
            torch.device("cpu"),
        )
        weighted, _ = _combined_loss(
            outputs,
            sample,
            nn.Identity(),
            nn.CrossEntropyLoss(
                weight=torch.tensor([1.0, 3.0]), reduction="none"
            ),
            nn.CrossEntropyLoss(reduction="none"),
            task_weights,
            torch.device("cpu"),
        )
        torch.testing.assert_close(weighted, unweighted * 3)


if __name__ == "__main__":
    unittest.main()
