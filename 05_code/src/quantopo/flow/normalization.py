"""Population normalization between native decoder and flow coordinates."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn

from quantopo.contracts.tensors import NativeRealization, NormalizedRealization, lengths_to_mask


class RealizationTransform(nn.Module):
    def __init__(self, mean: Tensor, scale: Tensor, *, scale_floor: float = 1e-4) -> None:
        super().__init__()
        if mean.shape != (128,) or scale.shape != (128,):
            raise ValueError("realization statistics must each have shape [128]")
        if scale_floor <= 0 or not bool(torch.isfinite(mean).all()):
            raise ValueError("realization statistics or scale floor are invalid")
        if not bool(torch.isfinite(scale).all()) or bool(torch.any(scale < 0)):
            raise ValueError("realization scale must be finite and nonnegative")
        self.scale_floor = scale_floor
        self.register_buffer("mean", mean.float().clone())
        self.register_buffer("scale", scale.float().clone().clamp_min(scale_floor))

    @classmethod
    def fit(
        cls, batches: Iterable[NativeRealization], *, scale_floor: float = 1e-4
    ) -> RealizationTransform:
        total = torch.zeros(128, dtype=torch.float64)
        squared = torch.zeros(128, dtype=torch.float64)
        count = 0
        for batch in batches:
            mask = lengths_to_mask(batch.lengths, batch.values.shape[1])
            valid = batch.values[mask].detach().to(device="cpu", dtype=torch.float64)
            total += valid.sum(dim=0)
            squared += valid.square().sum(dim=0)
            count += valid.shape[0]
        if count < 1:
            raise ValueError("cannot fit realization statistics without valid frames")
        mean = total / count
        variance = (squared / count - mean.square()).clamp_min(0.0)
        return cls(mean.float(), variance.sqrt().float(), scale_floor=scale_floor)

    def normalize(self, native: NativeRealization) -> NormalizedRealization:
        if not isinstance(native, NativeRealization):
            raise TypeError("normalize expects NativeRealization")
        mask = lengths_to_mask(native.lengths, native.values.shape[1])
        values = (native.values - self.mean.to(native.values)) / self.scale.to(native.values)
        values = values * mask.unsqueeze(-1).to(values.dtype)
        return NormalizedRealization(values, native.lengths)

    def denormalize(self, normalized: NormalizedRealization) -> NativeRealization:
        if not isinstance(normalized, NormalizedRealization):
            raise TypeError("denormalize expects NormalizedRealization")
        mask = lengths_to_mask(normalized.lengths, normalized.values.shape[1])
        values = (
            self.mean.to(normalized.values) + self.scale.to(normalized.values) * normalized.values
        )
        values = values * mask.unsqueeze(-1).to(values.dtype)
        return NativeRealization(values, normalized.lengths)
