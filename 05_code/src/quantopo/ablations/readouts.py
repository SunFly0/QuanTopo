"""Permutation-invariant and temporal plan descriptor readouts."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from quantopo.representation.plan_readout import PlanReadout


class EvaluationPlanReadout(nn.Module):
    """MLP used to measure descriptor information in a frozen plan representation."""

    def __init__(self, plan_dim: int = 256, descriptor_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(plan_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, descriptor_dim),
        )

    def forward(self, embeddings: Tensor, mask: Tensor) -> Tensor:
        if embeddings.ndim != 3 or mask.shape != embeddings.shape[:2]:
            raise ValueError("embeddings/mask must have shapes [B,L,D] and [B,L]")
        weight = mask.unsqueeze(-1).to(embeddings.dtype)
        pooled = (embeddings * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
        return cast(Tensor, self.network(pooled))


class TemporalPlanReadout(nn.Module):
    def __init__(
        self,
        plan_dim: int = 256,
        descriptor_dim: int = 128,
        *,
        width: int = 128,
        layers: int = 2,
        heads: int = 8,
        max_positions: int = 64,
    ) -> None:
        super().__init__()
        self.max_positions = max_positions
        self.input = nn.Linear(plan_dim, width)
        self.positions = nn.Parameter(torch.randn(max_positions, width) * 0.02)
        layer = nn.TransformerEncoderLayer(
            width,
            heads,
            dim_feedforward=4 * width,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=layers, norm=nn.LayerNorm(width), enable_nested_tensor=False
        )
        self.output = nn.Linear(width, descriptor_dim)

    def forward(self, embeddings: Tensor, mask: Tensor) -> Tensor:
        if embeddings.ndim != 3 or mask.shape != embeddings.shape[:2]:
            raise ValueError("embeddings/mask must have shapes [B,L,D] and [B,L]")
        if embeddings.shape[1] > self.max_positions:
            raise ValueError("plan exceeds temporal readout positions")
        hidden = self.input(embeddings) + self.positions[: embeddings.shape[1]]
        hidden = self.encoder(hidden, src_key_padding_mask=~mask)
        weight = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
        return cast(Tensor, self.output(pooled))


def build_plan_readout(name: str, *, plan_dim: int = 256, descriptor_dim: int = 128) -> nn.Module:
    if name == "masked_mean":
        return PlanReadout(plan_dim, descriptor_dim)
    if name == "temporal":
        return TemporalPlanReadout(plan_dim, descriptor_dim)
    if name == "evaluation_mlp":
        return EvaluationPlanReadout(plan_dim, descriptor_dim)
    raise KeyError(f"unknown plan readout control: {name}")
