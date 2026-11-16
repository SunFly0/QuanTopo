"""Exact zero-preserving geometric correction of planner logits."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn


class ZeroPreservingModulation(nn.Module):
    def __init__(
        self,
        *,
        state_width: int = 256,
        descriptor_dim: int = 128,
        hidden: int = 256,
        vocabulary_size: int = 512,
        alpha: float = 0.1,
        clip_value: float = 5.0,
        scale_floor: float = 1e-3,
    ) -> None:
        super().__init__()
        if min(state_width, descriptor_dim, hidden, vocabulary_size) < 1:
            raise ValueError("modulation dimensions must be positive")
        if alpha <= 0 or clip_value <= 0 or scale_floor <= 0:
            raise ValueError("modulation scales must be positive")
        self.state_width = state_width
        self.descriptor_dim = descriptor_dim
        self.vocabulary_size = vocabulary_size
        self.alpha = alpha
        self.clip_value = clip_value
        self.scale_floor = scale_floor
        self.network = nn.Sequential(
            nn.Linear(state_width + descriptor_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, vocabulary_size),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def _scaled_descriptor(self, descriptor: Tensor, scale: Tensor) -> Tensor:
        if descriptor.ndim != 2 or descriptor.shape[-1] != self.descriptor_dim:
            raise ValueError(f"descriptor must have shape [B,{self.descriptor_dim}]")
        if scale.shape not in ((self.descriptor_dim,), descriptor.shape):
            raise ValueError("descriptor scale must have shape [D] or [B,D]")
        return descriptor / scale.to(descriptor).clamp_min(self.scale_floor)

    def delta(self, state: Tensor, descriptor: Tensor, scale: Tensor) -> Tensor:
        if state.ndim != 3 or state.shape[-1] != self.state_width:
            raise ValueError(f"planner state must have shape [B,L,{self.state_width}]")
        if descriptor.shape[0] != state.shape[0]:
            raise ValueError("planner state and descriptor batch sizes must agree")
        scaled = self._scaled_descriptor(descriptor, scale)
        scaled = scaled.unsqueeze(1).expand(-1, state.shape[1], -1)
        zeros = torch.zeros_like(scaled)
        conditioned = torch.cat((state, scaled), dim=-1)
        baseline = torch.cat((state, zeros), dim=-1)
        paired = self.network(torch.cat((conditioned, baseline), dim=0))
        response, zero_response = paired.chunk(2, dim=0)
        delta = self.alpha * (response - zero_response).clamp(
            min=-self.clip_value, max=self.clip_value
        )
        return cast(Tensor, delta)

    def modulate(
        self, base_logits: Tensor, state: Tensor, descriptor: Tensor, scale: Tensor
    ) -> Tensor:
        if base_logits.shape != (*state.shape[:-1], self.vocabulary_size):
            raise ValueError(
                "base logits must have shape [B,L,vocabulary_size] matching planner state"
            )
        return base_logits + self.delta(state, descriptor, scale)
