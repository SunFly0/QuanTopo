"""Jointwise weighted whole-body readout and temporal prediction heads."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from quantopo.representation.temporal import TemporalResidualBlock


class WholeBodyReadout(nn.Module):
    def __init__(
        self,
        joints: int,
        physical_dim: int,
        frame_width: int,
        groups: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.joints = joints
        self.identity = nn.Parameter(torch.randn(joints, physical_dim) * 0.02)
        self.score = nn.Linear(physical_dim, 1)
        self.project = nn.Linear(physical_dim, frame_width)
        self.temporal = nn.ModuleList(
            TemporalResidualBlock(frame_width, groups, dilation, dropout) for dilation in (1, 2, 4)
        )
        self.root_head = nn.Sequential(
            nn.LayerNorm(frame_width),
            nn.Linear(frame_width, frame_width),
            nn.SiLU(),
            nn.Linear(frame_width, 4),
        )
        self.rotation_head = nn.Sequential(
            nn.LayerNorm(frame_width),
            nn.Linear(frame_width, frame_width),
            nn.SiLU(),
            nn.Linear(frame_width, (joints - 1) * 6),
        )

    def forward(self, joint_features: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        if joint_features.ndim != 4 or joint_features.shape[2] != self.joints:
            raise ValueError("joint features must have shape [B,T,J,D]")
        valid = mask[..., None, None].to(joint_features.dtype)
        features = joint_features * valid
        score = self.score(features + self.identity[None, None]).squeeze(-1)
        weights = torch.softmax(score, dim=-1)
        frame = (weights.unsqueeze(-1) * self.project(features)).sum(dim=2)
        temporal = frame.transpose(1, 2)
        for block in self.temporal:
            temporal = block(temporal, mask)
        temporal = temporal.transpose(1, 2)
        root = self.root_head(temporal) * mask.unsqueeze(-1).to(temporal.dtype)
        rotation = self.rotation_head(temporal).reshape(
            temporal.shape[0], temporal.shape[1], self.joints - 1, 6
        )
        rotation = rotation * mask[..., None, None].to(rotation.dtype)
        return root, rotation
