"""Stride-four plan encoder and plan-conditioned realization encoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from quantopo.contracts.tensors import NativeRealization, lengths_to_mask

from .quantizer import EMAQuantizer, QuantizedPlan
from .temporal import TemporalResidualBlock


def _apply_mask(values: Tensor, mask: Tensor) -> Tensor:
    return values * mask[:, None, :].to(values.dtype)


class PlanEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        width: int = 256,
        groups: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input = nn.Conv1d(input_dim, width, 5, padding=2)
        self.downsample = nn.ModuleList(
            (
                nn.Conv1d(width, width, 4, stride=2, padding=1),
                nn.Conv1d(width, width, 4, stride=2, padding=1),
            )
        )
        self.blocks = nn.ModuleList(
            TemporalResidualBlock(width, groups, dilation, dropout) for dilation in (1, 2, 4)
        )
        self.output_norm = nn.LayerNorm(width)

    def forward(self, ric: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        if ric.ndim != 3:
            raise ValueError("RIC input must have shape [B,T,D]")
        mask = lengths_to_mask(lengths, ric.shape[1])
        hidden = _apply_mask(self.input((ric * mask.unsqueeze(-1)).transpose(1, 2)), mask)
        current_lengths = lengths
        for convolution in self.downsample:
            hidden = convolution(hidden)
            current_lengths = torch.div(current_lengths, 2, rounding_mode="floor")
            mask = lengths_to_mask(current_lengths, hidden.shape[-1])
            hidden = _apply_mask(hidden, mask)
        expected = torch.div(lengths, 4, rounding_mode="floor")
        if not torch.equal(current_lengths, expected):
            raise RuntimeError("stride-four encoder violated floor-length contract")
        for block in self.blocks:
            hidden = block(hidden, mask)
        output = self.output_norm(hidden.transpose(1, 2))
        return output * mask.unsqueeze(-1).to(output.dtype), mask


def upsample_plan(
    plan: Tensor,
    plan_lengths: Tensor,
    frame_lengths: Tensor,
    *,
    frames: int,
) -> Tensor:
    """Linearly map each valid plan sequence to exactly its valid motion length."""

    if plan.ndim != 3 or plan_lengths.shape != frame_lengths.shape:
        raise ValueError("invalid plan or length shapes")
    output = plan.new_zeros(plan.shape[0], frames, plan.shape[-1])
    for batch_index in range(plan.shape[0]):
        plan_length = int(plan_lengths[batch_index].item())
        frame_length = int(frame_lengths[batch_index].item())
        if plan_length < 1 or frame_length < 1 or frame_length > frames:
            raise ValueError("invalid valid lengths for plan upsampling")
        source = plan[batch_index, :plan_length].transpose(0, 1).unsqueeze(0)
        if plan_length == 1:
            resized = source.expand(1, source.shape[1], frame_length)
        else:
            resized = functional.interpolate(
                source, size=frame_length, mode="linear", align_corners=True
            )
        output[batch_index, :frame_length] = resized[0].transpose(0, 1)
    return output


class RealizationEncoder(nn.Module):
    def __init__(
        self,
        ric_dim: int,
        plan_dim: int = 256,
        width: int = 256,
        groups: int = 32,
        dropout: float = 0.1,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        self.input = nn.Conv1d(ric_dim + plan_dim, width, 3, padding=1)
        self.blocks = nn.ModuleList(
            TemporalResidualBlock(width, groups, dilation, dropout) for dilation in (1, 2, 4, 8)
        )
        self.output = nn.Conv1d(width, output_dim, 1)

    def forward(
        self,
        ric: Tensor,
        frame_lengths: Tensor,
        plan_embeddings: Tensor,
        plan_lengths: Tensor,
    ) -> NativeRealization:
        mask = lengths_to_mask(frame_lengths, ric.shape[1])
        upsampled = upsample_plan(plan_embeddings, plan_lengths, frame_lengths, frames=ric.shape[1])
        combined = torch.cat((ric * mask.unsqueeze(-1), upsampled), dim=-1)
        hidden = _apply_mask(self.input(combined.transpose(1, 2)), mask)
        for block in self.blocks:
            hidden = block(hidden, mask)
        values = _apply_mask(self.output(hidden), mask).transpose(1, 2)
        return NativeRealization(values, frame_lengths)


@dataclass(frozen=True)
class EncodedMotion:
    quantized: QuantizedPlan
    native_realization: NativeRealization


def encode_motion(
    ric: Tensor,
    lengths: Tensor,
    plan_encoder: PlanEncoder,
    quantizer: EMAQuantizer,
    realization_encoder: RealizationEncoder,
) -> EncodedMotion:
    hidden, mask = plan_encoder(ric, lengths)
    quantized = quantizer(hidden, mask)
    native = realization_encoder(
        ric,
        lengths,
        quantized.embeddings,
        mask.sum(dim=1),
    )
    return EncodedMotion(quantized, native)
