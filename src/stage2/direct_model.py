"""Spatial-attention temporal model for the three directly labelled Stage 2 tasks."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


class SpatialAttentionPool(nn.Module):
    """Preserve local evidence while retaining a global scene summary."""

    def __init__(
        self,
        input_channels: int = 256,
        projection_size: int = 128,
        output_size: int = 256,
        dropout: float = 0.2,
        include_coordinates: bool = False,
    ) -> None:
        super().__init__()
        self.include_coordinates = include_coordinates
        self.project = nn.Sequential(
            nn.Conv2d(input_channels, projection_size, kernel_size=1),
            nn.GELU(),
        )
        self.attention = nn.Sequential(
            nn.Conv2d(projection_size, max(16, projection_size // 2), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(max(16, projection_size // 2), 1, kernel_size=1),
        )
        coordinate_size = 8 if include_coordinates else 0
        self.fuse = nn.Sequential(
            nn.Linear(projection_size * 2 + coordinate_size, output_size),
            nn.LayerNorm(output_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    @staticmethod
    def attention_coordinate_features(
        weights: Tensor,
        height: int,
        width: int,
    ) -> Tensor:
        """Summarize where each attention map is concentrated.

        The returned values are ``mean_x, mean_y, std_x, std_y`` followed by
        attention mass in the left, right, top and bottom halves. Coordinates
        use the resolution-independent range [-1, 1].
        """

        if weights.ndim != 3 or weights.shape[1] != 1:
            raise ValueError("weights must have shape [N, 1, H*W]")
        if height < 1 or width < 1 or weights.shape[2] != height * width:
            raise ValueError("Attention shape does not match height and width")
        y_axis = torch.linspace(
            -1.0, 1.0, height, device=weights.device, dtype=weights.dtype
        )
        x_axis = torch.linspace(
            -1.0, 1.0, width, device=weights.device, dtype=weights.dtype
        )
        yy, xx = torch.meshgrid(y_axis, x_axis, indexing="ij")
        xx = xx.flatten().unsqueeze(0)
        yy = yy.flatten().unsqueeze(0)
        flat_weights = weights.squeeze(1)

        mean_x = (flat_weights * xx).sum(dim=1)
        mean_y = (flat_weights * yy).sum(dim=1)
        std_x = torch.sqrt(
            (flat_weights * (xx - mean_x.unsqueeze(1)).square()).sum(dim=1)
            + 1e-6
        )
        std_y = torch.sqrt(
            (flat_weights * (yy - mean_y.unsqueeze(1)).square()).sum(dim=1)
            + 1e-6
        )
        left_mass = (flat_weights * (xx < 0).to(flat_weights.dtype)).sum(dim=1)
        right_mass = (flat_weights * (xx >= 0).to(flat_weights.dtype)).sum(dim=1)
        top_mass = (flat_weights * (yy < 0).to(flat_weights.dtype)).sum(dim=1)
        bottom_mass = (flat_weights * (yy >= 0).to(flat_weights.dtype)).sum(dim=1)
        return torch.stack(
            [
                mean_x,
                mean_y,
                std_x,
                std_y,
                left_mass,
                right_mass,
                top_mass,
                bottom_mass,
            ],
            dim=1,
        )

    def forward(self, maps: Tensor) -> tuple[Tensor, Tensor]:
        if maps.ndim != 5:
            raise ValueError("maps must have shape [B, T, C, H, W]")
        batch, steps, channels, height, width = maps.shape
        flat = maps.reshape(batch * steps, channels, height, width)
        projected = self.project(flat)
        weights = self.attention(projected).flatten(2).softmax(dim=-1)
        local = torch.bmm(projected.flatten(2), weights.transpose(1, 2)).squeeze(-1)
        global_scene = projected.mean(dim=(-2, -1))
        parts = [local, global_scene]
        if self.include_coordinates:
            parts.append(
                self.attention_coordinate_features(weights, height, width)
            )
        vectors = self.fuse(torch.cat(parts, dim=-1))
        return vectors.reshape(batch, steps, -1), weights.reshape(batch, steps, height, width)


class Stage2DirectSpatial(nn.Module):
    """Predict entry frame, entry side and evasion space from cached spatial maps.

    ``collision_indices`` are supplied by the collision branch at inference.
    During training they come from the directly annotated collision frame, with
    optional jitter in the training loop. A value of -1 means that no collision
    annotation exists, so the whole-video context is used instead.
    """

    def __init__(
        self,
        input_channels: int = 256,
        projection_size: int = 128,
        temporal_input_size: int = 256,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
        collision_window: int = 2,
        entry_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if collision_window < 0:
            raise ValueError("collision_window cannot be negative")
        if entry_temperature <= 0:
            raise ValueError("entry_temperature must be positive")
        self.collision_window = collision_window
        self.entry_temperature = entry_temperature
        self.spatial = SpatialAttentionPool(
            input_channels=input_channels,
            projection_size=projection_size,
            output_size=temporal_input_size,
            dropout=dropout,
        )
        self.temporal = nn.GRU(
            temporal_input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        dimension = hidden_size * 2
        self.dropout = nn.Dropout(dropout)
        self.entry_head = nn.Linear(dimension, 1)
        self.side_head = nn.Sequential(
            nn.Linear(dimension, hidden_size), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_size, 2)
        )
        self.evasion_head = nn.Sequential(
            nn.Linear(dimension * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )

    def _collision_context(self, hidden: Tensor, collision_indices: Tensor) -> Tensor:
        batch, steps, _ = hidden.shape
        indices = collision_indices.to(device=hidden.device, dtype=torch.long).reshape(-1)
        if len(indices) != batch or (indices < -1).any() or (indices >= steps).any():
            raise ValueError("collision_indices must contain an index or -1 per video")
        contexts = []
        for batch_index, center in enumerate(indices.tolist()):
            if center == -1:
                contexts.append(hidden[batch_index].mean(dim=0))
            else:
                start = max(0, center - self.collision_window)
                end = min(steps, center + self.collision_window + 1)
                contexts.append(hidden[batch_index, start:end].mean(dim=0))
        return torch.stack(contexts)

    def forward(
        self,
        maps: Tensor,
        collision_indices: Tensor,
        *,
        return_attention: bool = False,
    ) -> dict[str, Tensor]:
        vectors, attention = self.spatial(maps)
        hidden, _ = self.temporal(vectors)
        hidden = self.dropout(hidden)
        entry_logits = self.entry_head(hidden).squeeze(-1)

        entry_weights = (entry_logits / self.entry_temperature).softmax(dim=1)
        entry_context = torch.bmm(entry_weights.unsqueeze(1), hidden).squeeze(1)
        side_logits = self.side_head(entry_context)

        collision_context = self._collision_context(hidden, collision_indices)
        global_context = hidden.mean(dim=1)
        evasion_logits = self.evasion_head(torch.cat([collision_context, global_context], dim=-1))

        outputs = {
            "entry_logits": entry_logits,
            "side_logits": side_logits,
            "evasion_logits": evasion_logits,
        }
        if return_attention:
            outputs["spatial_attention"] = attention
            outputs["entry_attention"] = entry_weights
        return outputs


def direct_model_from_checkpoint(checkpoint: dict[str, Any]) -> Stage2DirectSpatial:
    model = Stage2DirectSpatial(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
