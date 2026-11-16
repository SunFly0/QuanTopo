"""Tensor structures and shape validation used across the model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

_DATASET_SHAPES = {
    "humanml3d": (22, 263, "HumanML3D"),
    "kitml": (21, 251, "KIT-ML"),
}


def _require_integer_lengths(lengths: Tensor, batch: int, frames: int) -> None:
    if lengths.ndim != 1 or lengths.shape[0] != batch:
        raise ValueError(f"lengths must have shape [{batch}]")
    if lengths.dtype not in (torch.int32, torch.int64):
        raise TypeError("lengths must use an integer dtype")
    if bool(torch.any(lengths < 1)) or bool(torch.any(lengths > frames)):
        raise ValueError(f"lengths must lie in [1,{frames}]")


def lengths_to_mask(lengths: Tensor, width: int) -> Tensor:
    """Return a boolean right-padding mask on the same device as ``lengths``."""

    return torch.arange(width, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


def _require_zero_padding(values: Tensor, lengths: Tensor, label: str) -> None:
    mask = lengths_to_mask(lengths, values.shape[1])
    padded = values.masked_select(~mask.unsqueeze(-1))
    if padded.numel() and bool(torch.any(padded != 0)):
        raise ValueError(f"padded {label} values must be zero")


@dataclass(frozen=True)
class NativeRealization:
    """Frame-rate realization in decoder-native coordinates."""

    values: Tensor
    lengths: Tensor

    def __post_init__(self) -> None:
        if self.values.ndim != 3 or self.values.shape[-1] != 128:
            raise ValueError("native realization must have shape [B,T,128]")
        _require_integer_lengths(self.lengths, self.values.shape[0], self.values.shape[1])
        _require_zero_padding(self.values, self.lengths, "realization")


@dataclass(frozen=True)
class NormalizedRealization:
    """Frame-rate realization after the frozen flow normalization."""

    values: Tensor
    lengths: Tensor

    def __post_init__(self) -> None:
        if self.values.ndim != 3 or self.values.shape[-1] != 128:
            raise ValueError("normalized realization must have shape [B,T,128]")
        _require_integer_lengths(self.lengths, self.values.shape[0], self.values.shape[1])
        _require_zero_padding(self.values, self.lengths, "realization")


def require_normalized(value: NormalizedRealization) -> Tensor:
    if not isinstance(value, NormalizedRealization):
        raise TypeError("expected NormalizedRealization")
    return value.values


@dataclass(frozen=True)
class PersistenceBatch:
    """Padded H0/H1 persistence points and their fixed readout."""

    points: Tensor
    mask: Tensor
    descriptor: Tensor

    def __post_init__(self) -> None:
        if self.points.ndim != 4 or self.points.shape[1] != 2 or self.points.shape[-1] != 2:
            raise ValueError("points must have shape [B,2,M,2]")
        if self.mask.shape != self.points.shape[:-1] or self.mask.dtype != torch.bool:
            raise ValueError("mask must be boolean with shape [B,2,M]")
        if self.descriptor.shape != (self.points.shape[0], 128):
            raise ValueError("descriptor must have shape [B,128]")
        if not bool(torch.isfinite(self.points).all()):
            raise ValueError("persistence points must be finite")
        if not bool(torch.isfinite(self.descriptor).all()):
            raise ValueError("persistence descriptor must be finite")


@dataclass(frozen=True)
class PlanBatch:
    """Right-padded discrete plan tokens."""

    indices: Tensor
    mask: Tensor
    lengths: Tensor
    codebook_size: int = 512

    def __post_init__(self) -> None:
        if self.indices.ndim != 2 or self.indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("plan indices must be an integer tensor [B,L]")
        _require_integer_lengths(self.lengths, self.indices.shape[0], self.indices.shape[1])
        expected_mask = lengths_to_mask(self.lengths, self.indices.shape[1])
        if self.mask.dtype != torch.bool or not torch.equal(self.mask, expected_mask):
            raise ValueError("plan mask must be the right-padding mask implied by lengths")
        if bool(torch.any(self.indices < 0)) or bool(torch.any(self.indices >= self.codebook_size)):
            raise ValueError(f"plan index outside codebook of size {self.codebook_size}")


@dataclass(frozen=True)
class MotionBatch:
    """A right-padded batch carrying both normalized RIC and physical joints."""

    dataset: str
    ric: Tensor
    joints: Tensor
    lengths: Tensor
    captions: tuple[str, ...]
    motion_ids: tuple[str, ...]
    frame_mask: Tensor
    plan_lengths: Tensor
    plan_mask: Tensor

    @classmethod
    def from_tensors(
        cls,
        *,
        dataset: str,
        ric: Tensor,
        joints: Tensor,
        lengths: Tensor,
        captions: Sequence[str],
        motion_ids: Sequence[str],
    ) -> MotionBatch:
        normalized_name = dataset.lower()
        if normalized_name not in _DATASET_SHAPES:
            raise ValueError(f"unsupported dataset: {dataset}")
        joint_count, ric_width, display_name = _DATASET_SHAPES[normalized_name]
        if ric.ndim != 3 or ric.shape[-1] != ric_width:
            raise ValueError(f"{display_name} RIC width must be {ric_width}")
        batch, frames, _ = ric.shape
        if joints.shape != (batch, frames, joint_count, 3):
            raise ValueError(f"{display_name} joints must have shape [B,T,{joint_count},3]")
        _require_integer_lengths(lengths, batch, frames)
        if len(captions) != batch or len(motion_ids) != batch:
            raise ValueError("captions and motion_ids must each contain B entries")
        if not bool(torch.isfinite(ric).all()) or not bool(torch.isfinite(joints).all()):
            raise ValueError("motion tensors must be finite")
        frame_mask = lengths_to_mask(lengths, frames)
        plan_lengths = torch.div(lengths, 4, rounding_mode="floor")
        if bool(torch.any(plan_lengths < 1)):
            raise ValueError("motion lengths must be at least four frames")
        plan_mask = lengths_to_mask(plan_lengths, int(plan_lengths.max().item()))
        return cls(
            dataset=normalized_name,
            ric=ric,
            joints=joints,
            lengths=lengths,
            captions=tuple(captions),
            motion_ids=tuple(motion_ids),
            frame_mask=frame_mask,
            plan_lengths=plan_lengths,
            plan_mask=plan_mask,
        )
