"""Root-motion recovery under the HumanML3D/KIT half-yaw convention."""

from __future__ import annotations

import torch
from torch import Tensor


def _validate_mask(mask: Tensor, batch: int, frames: int) -> None:
    if mask.shape != (batch, frames) or mask.dtype != torch.bool:
        raise ValueError(f"mask must be boolean with shape [{batch},{frames}]")
    transitions = mask[:, 1:].to(torch.int8) - mask[:, :-1].to(torch.int8)
    if bool(torch.any(transitions > 0)):
        raise ValueError("mask must describe right padding")


def recover_root_transform(
    root_channels: Tensor, mask: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Recover world-from-root rotations and positions from four RIC root channels.

    Channels are half-yaw increment, local horizontal velocity ``(vx,vz)``, and
    absolute root height. Padded outputs are exactly zero.
    """

    if root_channels.ndim != 3 or root_channels.shape[-1] != 4:
        raise ValueError("root channels must have shape [B,T,4]")
    batch, frames, _ = root_channels.shape
    if mask is None:
        mask = torch.ones(batch, frames, dtype=torch.bool, device=root_channels.device)
    _validate_mask(mask, batch, frames)
    valid = mask.unsqueeze(-1)
    channels = root_channels * valid.to(root_channels.dtype)

    theta = torch.zeros_like(channels[..., 0])
    if frames > 1:
        theta[:, 1:] = torch.cumsum(channels[:, :-1, 0], dim=1)
    angle = 2.0 * theta
    cosine, sine = torch.cos(angle), torch.sin(angle)
    rotation = torch.zeros(
        batch, frames, 3, 3, dtype=root_channels.dtype, device=root_channels.device
    )
    rotation[..., 0, 0] = cosine
    rotation[..., 0, 2] = -sine
    rotation[..., 1, 1] = 1.0
    rotation[..., 2, 0] = sine
    rotation[..., 2, 2] = cosine

    position = torch.zeros(batch, frames, 3, dtype=root_channels.dtype, device=root_channels.device)
    if frames > 1:
        local_step = torch.zeros_like(position[:, :-1])
        local_step[..., 0] = channels[:, :-1, 1]
        local_step[..., 2] = channels[:, :-1, 2]
        step = torch.einsum("btij,btj->bti", rotation[:, 1:], local_step)
        pair_mask = (mask[:, 1:] & mask[:, :-1]).unsqueeze(-1)
        position[:, 1:] = torch.cumsum(step * pair_mask.to(step.dtype), dim=1)
    position[..., 1] = channels[..., 3]

    rotation = rotation * mask[..., None, None].to(rotation.dtype)
    position = position * valid.to(position.dtype)
    return rotation, position
