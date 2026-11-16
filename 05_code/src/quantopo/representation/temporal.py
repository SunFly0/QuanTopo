"""Padding-aware temporal residual layers."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn


class MaskedGroupNorm(nn.Module):
    def __init__(self, channels: int, groups: int, epsilon: float = 1e-5) -> None:
        super().__init__()
        if channels % groups:
            raise ValueError("channels must be divisible by groups")
        self.channels = channels
        self.groups = groups
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        if values.ndim != 3 or values.shape[1] != self.channels:
            raise ValueError(f"values must have shape [B,{self.channels},T]")
        if mask.shape != (values.shape[0], values.shape[2]) or mask.dtype != torch.bool:
            raise ValueError("mask must have shape [B,T]")
        batch, _, frames = values.shape
        grouped = values.reshape(batch, self.groups, self.channels // self.groups, frames)
        valid = mask[:, None, None, :].to(values.dtype)
        denominator = (valid.sum(dim=-1, keepdim=True) * (self.channels // self.groups)).clamp_min(
            1
        )
        mean = (grouped * valid).sum(dim=(-2, -1), keepdim=True) / denominator
        variance = ((grouped - mean).square() * valid).sum(dim=(-2, -1), keepdim=True)
        variance = variance / denominator
        normalized = (grouped - mean) * torch.rsqrt(variance + self.epsilon)
        normalized = normalized.reshape_as(values)
        output = normalized * self.weight[None, :, None] + self.bias[None, :, None]
        return output * mask[:, None, :].to(output.dtype)


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, groups: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = MaskedGroupNorm(channels, groups)
        self.norm2 = MaskedGroupNorm(channels, groups)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        valid = mask[:, None, :].to(values.dtype)
        residual = values * valid
        hidden = self.conv1(torch.nn.functional.silu(self.norm1(residual, mask))) * valid
        hidden = self.dropout(hidden)
        hidden = self.conv2(torch.nn.functional.silu(self.norm2(hidden, mask))) * valid
        return cast(Tensor, (residual + self.dropout(hidden)) * valid)
