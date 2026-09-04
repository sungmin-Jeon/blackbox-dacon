from __future__ import annotations

import torch
from torch import nn
from torchvision.models.video import MViT_V2_S_Weights, mvit_v2_s


class Stage1MViT(nn.Module):
    def __init__(self, *, pretrained: bool = False, num_classes: int = 2) -> None:
        super().__init__()
        weights = MViT_V2_S_Weights.DEFAULT if pretrained else None
        self.net = mvit_v2_s(weights=weights)
        self.net.head[1] = nn.Linear(
            self.net.head[1].in_features,
            num_classes,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def build_stage1_model(*, pretrained: bool = True) -> Stage1MViT:
    """Build the two-class MViTv2-S used by Stage 1."""
    return Stage1MViT(pretrained=pretrained, num_classes=2)
