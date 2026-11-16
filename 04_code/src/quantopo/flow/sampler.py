"""Fixed-step Euler integration for conditional flow inference."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn


@torch.no_grad()
def euler_sample(
    vector_field: nn.Module,
    initial: Tensor,
    *,
    steps: int = 20,
    conditions: Mapping[str, object],
) -> Tensor:
    if steps < 1:
        raise ValueError("Euler integration requires at least one step")
    state = initial
    step_size = 1.0 / steps
    for index in range(steps):
        time = torch.full(
            (state.shape[0],),
            index * step_size,
            dtype=state.dtype,
            device=state.device,
        )
        state = state + step_size * vector_field(state, time, **conditions)
    return state
