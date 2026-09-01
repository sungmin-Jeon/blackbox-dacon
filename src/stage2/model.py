from __future__ import annotations

import torch
from torch import nn


class Stage2Temporal(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.r = nn.GRU(512, 192, 2, batch_first=True, bidirectional=True, dropout=0.15)
        self.tc = nn.Linear(384, 1)
        self.te = nn.Linear(384, 1)
        self.scene = nn.Sequential(nn.Linear(768, 192), nn.ReLU(), nn.Dropout(0.2), nn.Linear(192, 4))

    def logits(self, inputs: torch.Tensor):
        hidden, _ = self.r(inputs)
        return self.tc(hidden).squeeze(-1), self.te(hidden).squeeze(-1), hidden

    def forward(self, inputs: torch.Tensor):
        collision, entry, hidden = self.logits(inputs)
        collision_index = collision.argmax(1)
        entry_index = entry.argmax(1)
        batch = torch.arange(len(hidden), device=hidden.device)
        scene_input = torch.cat([hidden[batch, collision_index], hidden[batch, entry_index]], 1)
        return collision_index, entry_index, self.scene(scene_input)
