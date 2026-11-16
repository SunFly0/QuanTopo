"""Continuous 6D rotation representation with deterministic degeneracy handling."""

from __future__ import annotations

import torch
from torch import Tensor


def _safe_unit(vector: Tensor, fallback: Tensor, epsilon: float) -> Tensor:
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    normalized = vector / norm.clamp_min(epsilon)
    return torch.where(norm > epsilon, normalized, fallback.expand_as(vector))


def rotation_6d_to_matrix(rotation: Tensor, epsilon: float = 1e-6) -> Tensor:
    """Convert Zhou et al. 6D row representation to a proper rotation matrix.

    Zero and collinear inputs receive deterministic orthogonal fallbacks rather than
    producing NaNs. Regular inputs retain the usual Gram--Schmidt gradients.
    """

    if rotation.shape[-1] != 6:
        raise ValueError("6D rotations must have final dimension 6")
    first, second = rotation[..., :3], rotation[..., 3:]
    x_axis = torch.zeros_like(first)
    x_axis[..., 0] = 1.0
    b1 = _safe_unit(first, x_axis, epsilon)

    residual = second - (b1 * second).sum(dim=-1, keepdim=True) * b1
    candidate_x = torch.zeros_like(b1)
    candidate_x[..., 0] = 1.0
    candidate_y = torch.zeros_like(b1)
    candidate_y[..., 1] = 1.0
    candidate = torch.where((b1[..., :1].abs() < 0.9), candidate_x, candidate_y)
    fallback_residual = candidate - (candidate * b1).sum(dim=-1, keepdim=True) * b1
    fallback_b2 = _safe_unit(fallback_residual, candidate_y, epsilon)
    b2 = _safe_unit(residual, fallback_b2, epsilon)
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: Tensor) -> Tensor:
    """Return the first two rows of a rotation matrix as a 6D vector."""

    if matrix.shape[-2:] != (3, 3):
        raise ValueError("rotation matrices must have shape [...,3,3]")
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)
