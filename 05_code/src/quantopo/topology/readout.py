"""Fixed Gaussian persistence-image readout shared by references and predictions."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PersistenceReadout(nn.Module):
    def __init__(self, grid_size: int = 8, bandwidth: float = 0.12) -> None:
        super().__init__()
        if grid_size < 2 or bandwidth <= 0:
            raise ValueError("grid_size must be at least two and bandwidth positive")
        coordinates = torch.linspace(0.0, 1.0, grid_size)
        birth, lifetime = torch.meshgrid(coordinates, coordinates, indexing="ij")
        self.register_buffer("grid", torch.stack((birth, lifetime), dim=-1).reshape(-1, 2))
        self.grid_size = grid_size
        self.bandwidth = bandwidth

    def forward(self, points: Tensor, mass: Tensor) -> Tensor:
        if points.ndim != 4 or points.shape[1] != 2 or points.shape[-1] != 2:
            raise ValueError("points must have shape [B,2,M,2]")
        if mass.shape != points.shape[:-1]:
            raise ValueError("mass must have shape [B,2,M]")
        birth = points[..., 0]
        lifetime = (points[..., 1] - points[..., 0]).clamp_min(0.0)
        mapped = torch.stack((birth / (1.0 + birth), lifetime / (1.0 + lifetime)), dim=-1)
        squared_distance = (mapped.unsqueeze(-2) - self.grid).square().sum(dim=-1)
        weight = mass.to(points.dtype) * lifetime
        image = (weight.unsqueeze(-1) * torch.exp(-squared_distance / (2 * self.bandwidth**2))).sum(
            dim=-2
        )
        return image.flatten(start_dim=1)
