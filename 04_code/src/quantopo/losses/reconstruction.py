"""Sequence-balanced reconstruction losses in explicit coordinate systems."""

from __future__ import annotations

from torch import Tensor
from torch.nn import functional


def _per_sequence_mean(values: Tensor, mask: Tensor) -> Tensor:
    expanded = mask
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    weight = expanded.to(values.dtype).expand_as(values)
    numerator = (values * weight).flatten(start_dim=1).sum(dim=1)
    denominator = weight.flatten(start_dim=1).sum(dim=1).clamp_min(1.0)
    return (numerator / denominator).mean()


def motion_reconstruction_loss(
    predicted_normalized_ric: Tensor,
    target_normalized_ric: Tensor,
    predicted_joints: Tensor,
    target_joints: Tensor,
    mask: Tensor,
    bone_reference: Tensor,
) -> Tensor:
    if predicted_normalized_ric.shape != target_normalized_ric.shape:
        raise ValueError("predicted and target RIC shapes differ")
    if predicted_joints.shape != target_joints.shape:
        raise ValueError("predicted and target joint shapes differ")
    ric_error = functional.smooth_l1_loss(
        predicted_normalized_ric, target_normalized_ric, reduction="none"
    )
    joint_error = (predicted_joints - target_joints).abs() / bone_reference.clamp_min(1e-6)
    return _per_sequence_mean(ric_error, mask) + _per_sequence_mean(joint_error, mask)


def physical_velocity_loss(
    predicted_joints: Tensor,
    target_joints: Tensor,
    mask: Tensor,
    bone_reference: Tensor,
) -> Tensor:
    if predicted_joints.shape != target_joints.shape:
        raise ValueError("predicted and target joint shapes differ")
    if predicted_joints.shape[1] < 2:
        return predicted_joints.sum() * 0.0
    predicted_velocity = predicted_joints[:, 1:] - predicted_joints[:, :-1]
    target_velocity = target_joints[:, 1:] - target_joints[:, :-1]
    pair_mask = mask[:, 1:] & mask[:, :-1]
    error = (predicted_velocity - target_velocity).abs() / bone_reference.clamp_min(1e-6)
    return _per_sequence_mean(error, pair_mask)
