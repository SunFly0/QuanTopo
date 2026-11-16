"""Trunk-centered anatomical query graph used by the topology predictor."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TrunkStarMessagePassing(nn.Module):
    """Two-way gated messages between the trunk query and five part queries."""

    def __init__(self, width: int = 512, rounds: int = 2) -> None:
        super().__init__()
        if width < 1 or rounds < 1:
            raise ValueError("width and rounds must be positive")
        self.rounds = rounds
        self.trunk_messages = nn.ModuleList(nn.Linear(width, width) for _ in range(rounds))
        self.part_messages = nn.ModuleList(nn.Linear(width, width) for _ in range(rounds))
        self.trunk_gates = nn.ModuleList(nn.Linear(2 * width, width) for _ in range(rounds))
        self.part_gates = nn.ModuleList(nn.Linear(2 * width, width) for _ in range(rounds))
        self.normalizers = nn.ModuleList(nn.LayerNorm(width) for _ in range(rounds))

    def forward(self, queries: Tensor) -> Tensor:
        if queries.ndim != 3 or queries.shape[1] != 6:
            raise ValueError("anatomical queries must have shape [B,6,D]")
        state = queries
        for index in range(self.rounds):
            trunk = state[:, :1]
            parts = state[:, 1:]
            incoming_trunk = self.trunk_messages[index](parts).mean(dim=1, keepdim=True)
            incoming_parts = self.part_messages[index](trunk).expand_as(parts)
            trunk_gate = torch.sigmoid(
                self.trunk_gates[index](torch.cat((trunk, incoming_trunk), dim=-1))
            )
            part_gate = torch.sigmoid(
                self.part_gates[index](torch.cat((parts, incoming_parts), dim=-1))
            )
            updated = torch.cat(
                (
                    trunk + trunk_gate * incoming_trunk,
                    parts + part_gate * incoming_parts,
                ),
                dim=1,
            )
            state = self.normalizers[index](updated)
        return state
