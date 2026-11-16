"""Six-part plan routing, joint construction, readout, and kinematic decoding."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from quantopo.contracts.tensors import NativeRealization, lengths_to_mask
from quantopo.data.specs import DatasetSpec
from quantopo.geometry.ric import assemble_ric
from quantopo.geometry.rotations import rotation_6d_to_matrix
from quantopo.representation.encoders import upsample_plan

from .mps import ConditionalMPS
from .readout import WholeBodyReadout
from .routing import anatomical_part_indices, depth_first_order


@dataclass(frozen=True)
class DecodedMotion:
    ric: Tensor
    joints: Tensor
    root_channels: Tensor
    local_rotations: Tensor


class AnatomicalDecoder(nn.Module):
    def __init__(
        self,
        spec: DatasetSpec,
        offsets: Tensor,
        *,
        plan_dim: int = 256,
        realization_dim: int = 128,
        physical_dim: int = 384,
        part_bond_dim: int = 8,
        joint_bond_dim: int = 16,
        identity_dim: int = 256,
        frame_width: int = 384,
        groups: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if realization_dim != 128:
            raise ValueError("realization dimension must be 128")
        if offsets.shape != (spec.joint_count, 3):
            raise ValueError(f"offsets must have shape [{spec.joint_count},3]")
        self.spec = spec
        self.plan_dim = plan_dim
        self.realization_dim = realization_dim
        self.register_buffer("offsets", offsets.to(torch.float32).clone())
        self.register_buffer(
            "part_for_joint", torch.tensor(anatomical_part_indices(spec), dtype=torch.long)
        )
        traversal = depth_first_order(spec.parents)
        self.register_buffer("traversal", torch.tensor(traversal, dtype=torch.long))
        inverse = [0] * spec.joint_count
        for site, joint in enumerate(traversal):
            inverse[joint] = site
        self.register_buffer("inverse_traversal", torch.tensor(inverse, dtype=torch.long))

        self.part_identity = nn.Embedding(6, plan_dim)
        self.joint_identity = nn.Embedding(spec.joint_count, identity_dim)
        self.part_mps = ConditionalMPS(6, physical_dim, part_bond_dim, 2 * plan_dim)
        joint_condition_dim = realization_dim + physical_dim + identity_dim
        self.joint_mps = ConditionalMPS(
            spec.joint_count, physical_dim, joint_bond_dim, joint_condition_dim
        )
        self.readout = WholeBodyReadout(
            spec.joint_count, physical_dim, frame_width, groups, dropout
        )

    def forward(
        self,
        plan_embeddings: Tensor,
        plan_lengths: Tensor,
        realization: NativeRealization,
    ) -> DecodedMotion:
        if not isinstance(realization, NativeRealization):
            raise TypeError("decoder expects NativeRealization")
        values = realization.values
        lengths = realization.lengths
        if (
            plan_embeddings.shape[0] != values.shape[0]
            or plan_embeddings.shape[-1] != self.plan_dim
        ):
            raise ValueError("plan embeddings have incompatible shape")
        mask = lengths_to_mask(lengths, values.shape[1])
        plan = upsample_plan(plan_embeddings, plan_lengths, lengths, frames=values.shape[1])
        part_id = self.part_identity.weight[None, None].expand(
            values.shape[0], values.shape[1], -1, -1
        )
        part_condition = torch.cat((plan.unsqueeze(2).expand(-1, -1, 6, -1), part_id), dim=-1)
        part_states = self.part_mps(part_condition)

        routed = part_states[:, :, self.part_for_joint]
        joint_id = self.joint_identity.weight[None, None].expand(
            values.shape[0], values.shape[1], -1, -1
        )
        joint_condition = torch.cat(
            (
                values.unsqueeze(2).expand(-1, -1, self.spec.joint_count, -1),
                routed,
                joint_id,
            ),
            dim=-1,
        )
        traversal_condition = joint_condition[:, :, self.traversal]
        traversal_features = self.joint_mps(traversal_condition)
        joint_features = traversal_features[:, :, self.inverse_traversal]
        root_channels, rotation_6d = self.readout(joint_features, mask)

        non_root = rotation_6d_to_matrix(rotation_6d)
        identity = torch.eye(3, dtype=non_root.dtype, device=non_root.device)
        root_rotation = identity.expand(values.shape[0], values.shape[1], 1, 3, 3)
        local_rotations = torch.cat((root_rotation, non_root), dim=2)
        local_rotations = local_rotations * mask[..., None, None, None].to(local_rotations.dtype)
        ric, joints = assemble_ric(
            local_rotations,
            root_channels,
            self.offsets.to(local_rotations.dtype),
            self.spec,
            mask,
        )
        return DecodedMotion(ric, joints, root_channels, local_rotations)
