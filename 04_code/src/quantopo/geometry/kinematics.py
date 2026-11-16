"""Fixed-offset forward kinematics."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def forward_kinematics(
    local_rotations: Tensor,
    offsets: Tensor,
    parents: Sequence[int],
    *,
    root_rotation: Tensor | None = None,
    root_position: Tensor | None = None,
    mask: Tensor | None = None,
) -> Tensor:
    """Apply recursive local rotations and fixed canonical offsets."""

    if local_rotations.ndim != 5 or local_rotations.shape[-2:] != (3, 3):
        raise ValueError("local rotations must have shape [B,T,J,3,3]")
    batch, frames, joints = local_rotations.shape[:3]
    if offsets.shape != (joints, 3):
        raise ValueError(f"offsets must have shape [{joints},3]")
    if len(parents) != joints or parents[0] != -1:
        raise ValueError("parents must define a root-first skeleton")
    for joint, parent in enumerate(parents[1:], 1):
        if parent < 0 or parent >= joint:
            raise ValueError(f"joint {joint} must follow its parent in traversal order")

    if root_rotation is None:
        root_rotation = torch.eye(
            3, dtype=local_rotations.dtype, device=local_rotations.device
        ).expand(batch, frames, 3, 3)
    if root_position is None:
        root_position = torch.zeros(
            batch, frames, 3, dtype=local_rotations.dtype, device=local_rotations.device
        )
    if root_rotation.shape != (batch, frames, 3, 3):
        raise ValueError("root_rotation must have shape [B,T,3,3]")
    if root_position.shape != (batch, frames, 3):
        raise ValueError("root_position must have shape [B,T,3]")

    global_rotations: list[Tensor] = [root_rotation]
    positions: list[Tensor] = [root_position]
    for joint in range(1, joints):
        parent = parents[joint]
        parent_rotation = global_rotations[parent]
        global_rotations.append(parent_rotation @ local_rotations[..., joint, :, :])
        bone = torch.einsum("btij,j->bti", parent_rotation, offsets[joint])
        positions.append(positions[parent] + bone)
    output = torch.stack(positions, dim=2)
    if mask is not None:
        if mask.shape != (batch, frames) or mask.dtype != torch.bool:
            raise ValueError(f"mask must be boolean with shape [{batch},{frames}]")
        output = output * mask[..., None, None].to(output.dtype)
    return output
