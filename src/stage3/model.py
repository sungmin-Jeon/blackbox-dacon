from __future__ import annotations

import torch
from torch import nn
from torchvision.models.video import mvit_v2_s


class Stage3MViT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = mvit_v2_s(weights=None)
        dimension = self.backbone.head[1].in_features
        self.backbone.head = nn.Identity()
        self.accel = nn.Linear(dimension, 4)
        self.steer = nn.Linear(dimension, 3)

    def forward(self, inputs: torch.Tensor):
        features = self.backbone(inputs)
        return self.accel(features), self.steer(features)
