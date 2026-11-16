"""Permutation-invariant descriptor readout from valid plan embeddings."""

from __future__ import annotations

from typing import cast

from torch import Tensor, nn


class PlanReadout(nn.Module):
    def __init__(self, plan_dim: int = 256, descriptor_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(plan_dim, plan_dim),
            nn.SiLU(),
            nn.Linear(plan_dim, descriptor_dim),
        )

    def forward(self, embeddings: Tensor, mask: Tensor) -> Tensor:
        if embeddings.ndim != 3 or mask.shape != embeddings.shape[:2]:
            raise ValueError("embeddings/mask must have shapes [B,L,D] and [B,L]")
        weight = mask.unsqueeze(-1).to(embeddings.dtype)
        pooled = (embeddings * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
        return cast(Tensor, self.network(pooled))
