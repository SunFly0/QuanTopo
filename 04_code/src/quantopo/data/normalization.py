"""Streaming train-only population statistics with explicit padding masks."""

from __future__ import annotations

import torch
from torch import Tensor


class RunningMoments:
    def __init__(self, width: int) -> None:
        if width <= 0:
            raise ValueError("width must be positive")
        self.width = int(width)
        self.count = 0
        self._mean = torch.zeros(width, dtype=torch.float64)
        self._m2 = torch.zeros(width, dtype=torch.float64)

    def update(self, values: Tensor, mask: Tensor) -> None:
        if values.ndim != 3 or values.shape[-1] != self.width:
            raise ValueError(f"values must have shape [B,T,{self.width}]")
        if mask.shape != values.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("mask must be boolean with shape [B,T]")
        valid = values.detach().to(dtype=torch.float64, device="cpu")[mask.to(device="cpu")]
        if valid.numel() == 0:
            return
        if not bool(torch.isfinite(valid).all()):
            raise ValueError("statistics input must be finite")
        batch_count = valid.shape[0]
        batch_mean = valid.mean(dim=0)
        centered = valid - batch_mean
        batch_m2 = (centered * centered).sum(dim=0)
        if self.count == 0:
            self._mean = batch_mean
            self._m2 = batch_m2
            self.count = batch_count
            return
        total = self.count + batch_count
        delta = batch_mean - self._mean
        self._mean = self._mean + delta * (batch_count / total)
        self._m2 = self._m2 + batch_m2 + delta.square() * (self.count * batch_count / total)
        self.count = total

    def finalize(self, *, std_floor: float) -> tuple[Tensor, Tensor]:
        if self.count == 0:
            raise ValueError("cannot finalize empty statistics")
        if std_floor <= 0:
            raise ValueError("std_floor must be positive")
        variance = self._m2 / self.count
        std = torch.sqrt(torch.clamp_min(variance, 0.0)).clamp_min(std_floor)
        return self._mean.clone(), std
