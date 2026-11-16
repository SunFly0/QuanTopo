"""Trajectory-state construction, sampling, and shared RMS normalization."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class TrajectorySample:
    raw_states: Tensor
    normalized_states: Tensor
    indices: Tensor
    mask: Tensor


def normalize_states(states: Tensor, epsilon: float = 1e-6) -> Tensor:
    """Center coordinates and divide all state blocks by one scalar RMS."""

    if states.ndim != 2 or states.shape[0] < 1:
        raise ValueError("states must have shape [N,D] with N >= 1")
    centered = states - states.mean(dim=0, keepdim=True)
    rms = centered.square().mean().sqrt().clamp_min(epsilon)
    return centered / rms


def normalize_states_block_balanced(
    states: Tensor, block_scales: Tensor, epsilon: float = 1e-6
) -> Tensor:
    """Apply frozen pose/root/displacement scales before whole-state normalization."""

    if states.ndim != 2 or states.shape[0] < 1 or (states.shape[1] - 3) % 6:
        raise ValueError("block-balanced states must have width 6J+3")
    if block_scales.shape != (3,):
        raise ValueError("block_scales must contain pose, root, and displacement scales")
    if not bool(torch.isfinite(block_scales).all()) or bool(torch.any(block_scales <= 0)):
        raise ValueError("block_scales must be finite and positive")
    joints = (states.shape[1] - 3) // 6
    pose_end = 3 * joints
    root_end = pose_end + 3
    boundaries = ((0, pose_end), (pose_end, root_end), (root_end, states.shape[1]))
    balanced = states.clone()
    for scale, (start, end) in zip(block_scales.to(states), boundaries, strict=True):
        balanced[:, start:end] /= scale.clamp_min(epsilon)
    return normalize_states(balanced, epsilon)


def _sample_indices(length: int, count: int, device: torch.device) -> Tensor:
    selected = min(length, count)
    if selected == length:
        return torch.arange(length, device=device)
    return torch.linspace(0, length - 1, selected, device=device).round().to(torch.long)


def build_trajectory_states(
    joints: Tensor, mask: Tensor, sample_count: int = 32
) -> TrajectorySample:
    """Build ``[root-relative pose, root, displacement]`` and sample valid states."""

    if joints.ndim != 4 or joints.shape[-1] != 3:
        raise ValueError("joints must have shape [B,T,J,3]")
    batch, frames, joint_count, _ = joints.shape
    if mask.shape != (batch, frames) or mask.dtype != torch.bool:
        raise ValueError(f"mask must be boolean with shape [{batch},{frames}]")
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if bool(torch.any(mask[:, 1:] & ~mask[:, :-1])):
        raise ValueError("mask must describe right padding")

    sample_lengths = torch.minimum(mask.sum(dim=1), torch.tensor(sample_count, device=mask.device))
    if bool(torch.any(sample_lengths < 1)):
        raise ValueError("each sequence must contain at least one valid frame")
    maximum = int(sample_lengths.max().item())
    state_width = 6 * joint_count + 3
    raw = joints.new_zeros(batch, maximum, state_width)
    normalized = torch.zeros_like(raw)
    indices = torch.full((batch, maximum), -1, dtype=torch.long, device=joints.device)
    sample_mask = torch.zeros(batch, maximum, dtype=torch.bool, device=joints.device)

    for batch_index in range(batch):
        length = int(mask[batch_index].sum().item())
        motion = joints[batch_index, :length]
        root = motion[:, :1]
        pose = motion - root
        displacement = torch.zeros_like(motion)
        if length > 1:
            displacement[:-1] = motion[1:] - motion[:-1]
            displacement[-1] = motion[-1] - motion[-2]
        states = torch.cat(
            (pose.flatten(start_dim=1), root[:, 0], displacement.flatten(start_dim=1)), dim=-1
        )
        chosen = _sample_indices(length, sample_count, joints.device)
        selected = states[chosen]
        count = chosen.numel()
        raw[batch_index, :count] = selected
        normalized[batch_index, :count] = normalize_states(selected)
        indices[batch_index, :count] = chosen
        sample_mask[batch_index, :count] = True
    return TrajectorySample(raw, normalized, indices, sample_mask)
