"""Assembly and physical-joint recovery for the ``12J-1`` RIC layout."""

from __future__ import annotations

import torch
from torch import Tensor

from quantopo.data.specs import DatasetSpec

from .kinematics import forward_kinematics
from .root import recover_root_transform
from .rotations import matrix_to_rotation_6d


def _require_motion_mask(mask: Tensor, batch: int, frames: int) -> None:
    if mask.shape != (batch, frames) or mask.dtype != torch.bool:
        raise ValueError(f"mask must be boolean with shape [{batch},{frames}]")


def _forward_difference(values: Tensor, mask: Tensor) -> Tensor:
    difference = torch.zeros_like(values)
    if values.shape[1] > 1:
        pair_mask = mask[:, 1:] & mask[:, :-1]
        difference[:, :-1] = (values[:, 1:] - values[:, :-1]) * pair_mask[..., None, None].to(
            values.dtype
        )
        lengths = mask.sum(dim=1)
        for batch_index, length_tensor in enumerate(lengths):
            length = int(length_tensor.item())
            if length > 1:
                difference[batch_index, length - 1] = difference[batch_index, length - 2]
    return difference


def recover_joints_from_ric(ric: Tensor, spec: DatasetSpec, mask: Tensor | None = None) -> Tensor:
    """Recover world-space joints from unnormalized dataset RIC features."""

    if ric.ndim != 3 or ric.shape[-1] != spec.ric_width:
        raise ValueError(f"{spec.display_name} RIC width must be {spec.ric_width}")
    batch, frames, _ = ric.shape
    if mask is None:
        mask = torch.ones(batch, frames, dtype=torch.bool, device=ric.device)
    _require_motion_mask(mask, batch, frames)
    clean = ric * mask.unsqueeze(-1).to(ric.dtype)
    root_rotation, root_position = recover_root_transform(clean[..., :4], mask)
    position_end = 4 + 3 * (spec.joint_count - 1)
    local = clean[..., 4:position_end].reshape(batch, frames, spec.joint_count - 1, 3)
    world = torch.einsum("btij,btkj->btki", root_rotation, local)
    world[..., 0] += root_position[..., None, 0]
    world[..., 2] += root_position[..., None, 2]
    joints = torch.cat((root_position.unsqueeze(2), world), dim=2)
    return joints * mask[..., None, None].to(joints.dtype)


def assemble_ric(
    local_rotations: Tensor,
    root_channels: Tensor,
    offsets: Tensor,
    spec: DatasetSpec,
    mask: Tensor | None = None,
    *,
    contact_height: tuple[float, float] = (0.08, 0.05),
    contact_velocity: tuple[float, float] = (0.002, 0.005),
    contact_temperature: float = 0.01,
) -> tuple[Tensor, Tensor]:
    """Assemble unnormalized RIC features and return their physical FK joints."""

    if local_rotations.ndim != 5 or local_rotations.shape[-2:] != (3, 3):
        raise ValueError("local rotations must have shape [B,T,J,3,3]")
    batch, frames, joints = local_rotations.shape[:3]
    if joints != spec.joint_count:
        raise ValueError(f"{spec.display_name} requires {spec.joint_count} joints")
    if root_channels.shape != (batch, frames, 4):
        raise ValueError("root_channels must have shape [B,T,4]")
    if mask is None:
        mask = torch.ones(batch, frames, dtype=torch.bool, device=local_rotations.device)
    _require_motion_mask(mask, batch, frames)
    if contact_temperature <= 0:
        raise ValueError("contact_temperature must be positive")

    root_channels = root_channels * mask.unsqueeze(-1).to(root_channels.dtype)
    root_rotation, root_position = recover_root_transform(root_channels, mask)
    physical = forward_kinematics(
        local_rotations,
        offsets,
        spec.parents,
        root_rotation=root_rotation,
        root_position=root_position,
        mask=mask,
    )

    shifted = physical[..., 1:, :].clone()
    shifted[..., 0] -= root_position[..., None, 0]
    shifted[..., 2] -= root_position[..., None, 2]
    local_positions = torch.einsum("btji,btkj->btki", root_rotation, shifted)
    rotation_6d = matrix_to_rotation_6d(local_rotations[..., 1:, :, :])

    world_velocity = _forward_difference(physical, mask)
    local_velocity = torch.einsum("btji,btkj->btki", root_rotation, world_velocity)

    foot = physical[..., list(spec.foot_indices), :]
    foot_delta = _forward_difference(foot, mask)
    squared_speed = foot_delta.square().sum(dim=-1)
    height_threshold = torch.tensor(
        [contact_height[0], contact_height[0], contact_height[1], contact_height[1]],
        dtype=physical.dtype,
        device=physical.device,
    )
    velocity_threshold = torch.tensor(
        [contact_velocity[0], contact_velocity[0], contact_velocity[1], contact_velocity[1]],
        dtype=physical.dtype,
        device=physical.device,
    )
    low = torch.sigmoid((height_threshold - foot[..., 1]) / contact_temperature)
    still = torch.sigmoid((velocity_threshold - squared_speed) / contact_temperature)
    contacts = low * still

    ric = torch.cat(
        (
            root_channels,
            local_positions.flatten(start_dim=-2),
            rotation_6d.flatten(start_dim=-2),
            local_velocity.flatten(start_dim=-2),
            contacts,
        ),
        dim=-1,
    )
    if ric.shape[-1] != spec.ric_width:
        raise RuntimeError(f"assembled RIC has width {ric.shape[-1]}, expected {spec.ric_width}")
    ric = ric * mask.unsqueeze(-1).to(ric.dtype)
    return ric, physical
