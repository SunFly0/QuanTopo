"""Decoder and deterministic-realization controls with full-model output contracts."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from quantopo.contracts.tensors import NativeRealization, NormalizedRealization, lengths_to_mask
from quantopo.data.specs import DatasetSpec
from quantopo.decoder.anatomical import AnatomicalDecoder, DecodedMotion
from quantopo.decoder.mps import ConditionalMPS
from quantopo.decoder.readout import WholeBodyReadout
from quantopo.decoder.routing import anatomical_part_indices, depth_first_order
from quantopo.flow.transformer import ConditionalFlow
from quantopo.geometry.ric import assemble_ric
from quantopo.geometry.rotations import rotation_6d_to_matrix
from quantopo.representation.encoders import upsample_plan
from quantopo.text.clip import TextFeatures


class _DecoderControl(nn.Module):
    def __init__(
        self,
        spec: DatasetSpec,
        offsets: Tensor,
        *,
        physical_dim: int,
        frame_width: int,
        groups: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if offsets.shape != (spec.joint_count, 3):
            raise ValueError(f"offsets must have shape [{spec.joint_count},3]")
        self.spec = spec
        self.register_buffer("offsets", offsets.float().clone())
        self.readout = WholeBodyReadout(
            spec.joint_count, physical_dim, frame_width, groups, dropout
        )

    def _motion(self, joint_features: Tensor, lengths: Tensor) -> DecodedMotion:
        mask = lengths_to_mask(lengths, joint_features.shape[1])
        root_channels, rotation_6d = self.readout(joint_features, mask)
        non_root = rotation_6d_to_matrix(rotation_6d)
        identity = torch.eye(3, dtype=non_root.dtype, device=non_root.device)
        root = identity.expand(joint_features.shape[0], joint_features.shape[1], 1, 3, 3)
        rotations = torch.cat((root, non_root), dim=2)
        rotations = rotations * mask[..., None, None, None].to(rotations.dtype)
        ric, joints = assemble_ric(rotations, root_channels, self.offsets, self.spec, mask)
        return DecodedMotion(ric, joints, root_channels, rotations)


class SharedContextDecoder(_DecoderControl):
    """Broadcast one plan context before joint-specific synthesis."""

    def __init__(
        self,
        spec: DatasetSpec,
        offsets: Tensor,
        *,
        plan_dim: int = 256,
        realization_dim: int = 128,
        physical_dim: int = 384,
        identity_dim: int = 256,
        frame_width: int = 384,
        groups: int = 32,
        dropout: float = 0.1,
        **_: object,
    ) -> None:
        super().__init__(
            spec,
            offsets,
            physical_dim=physical_dim,
            frame_width=frame_width,
            groups=groups,
            dropout=dropout,
        )
        self.plan_dim = plan_dim
        self.plan_projection = nn.Linear(plan_dim, physical_dim)
        self.joint_identity = nn.Embedding(spec.joint_count, identity_dim)
        self.joint_operator = nn.Sequential(
            nn.Linear(realization_dim + physical_dim + identity_dim, physical_dim),
            nn.SiLU(),
            nn.Linear(physical_dim, physical_dim),
            nn.LayerNorm(physical_dim),
        )

    def forward(
        self, plan_embeddings: Tensor, plan_lengths: Tensor, realization: NativeRealization
    ) -> DecodedMotion:
        values = realization.values
        plan = upsample_plan(
            plan_embeddings, plan_lengths, realization.lengths, frames=values.shape[1]
        )
        context = self.plan_projection(plan).unsqueeze(2).expand(-1, -1, self.spec.joint_count, -1)
        identity = self.joint_identity.weight[None, None].expand(
            values.shape[0], values.shape[1], -1, -1
        )
        joint_input = torch.cat(
            (
                values.unsqueeze(2).expand(-1, -1, self.spec.joint_count, -1),
                context,
                identity,
            ),
            dim=-1,
        )
        return self._motion(self.joint_operator(joint_input), realization.lengths)


class FlatMPSDecoder(_DecoderControl):
    """Remove the part hierarchy and contract one joint-chain conditional MPS."""

    def __init__(
        self,
        spec: DatasetSpec,
        offsets: Tensor,
        *,
        plan_dim: int = 256,
        realization_dim: int = 128,
        physical_dim: int = 384,
        joint_bond_dim: int = 16,
        identity_dim: int = 256,
        frame_width: int = 384,
        groups: int = 32,
        dropout: float = 0.1,
        **_: object,
    ) -> None:
        super().__init__(
            spec,
            offsets,
            physical_dim=physical_dim,
            frame_width=frame_width,
            groups=groups,
            dropout=dropout,
        )
        self.plan_dim = plan_dim
        self.joint_identity = nn.Embedding(spec.joint_count, identity_dim)
        self.register_buffer("traversal", torch.tensor(depth_first_order(spec.parents)))
        inverse = torch.empty(spec.joint_count, dtype=torch.long)
        inverse[self.traversal] = torch.arange(spec.joint_count)
        self.register_buffer("inverse_traversal", inverse)
        condition_dim = realization_dim + plan_dim + identity_dim
        self.joint_mps = ConditionalMPS(
            spec.joint_count, physical_dim, joint_bond_dim, condition_dim
        )

    def forward(
        self, plan_embeddings: Tensor, plan_lengths: Tensor, realization: NativeRealization
    ) -> DecodedMotion:
        values = realization.values
        plan = upsample_plan(
            plan_embeddings, plan_lengths, realization.lengths, frames=values.shape[1]
        )
        identity = self.joint_identity.weight[None, None].expand(
            values.shape[0], values.shape[1], -1, -1
        )
        condition = torch.cat(
            (
                values.unsqueeze(2).expand(-1, -1, self.spec.joint_count, -1),
                plan.unsqueeze(2).expand(-1, -1, self.spec.joint_count, -1),
                identity,
            ),
            dim=-1,
        )
        features = self.joint_mps(condition[:, :, self.traversal])
        return self._motion(features[:, :, self.inverse_traversal], realization.lengths)


def _mps_parameter_count(sites: int, physical_dim: int, bond_dim: int, condition_dim: int) -> int:
    bonds = [1] + [bond_dim] * (sites - 1) + [1]
    cores = sum(bonds[site] * physical_dim * bonds[site + 1] for site in range(sites))
    probes = sites * physical_dim
    conditioners = sites * (condition_dim * 2 * physical_dim + 2 * physical_dim)
    normalizers = sites * 2 * physical_dim
    return cores + probes + conditioners + normalizers


class _SitewiseConditionalMLP(nn.Module):
    def __init__(
        self,
        sites: int,
        input_dim: int,
        output_dim: int,
        *,
        target_parameters: int,
    ) -> None:
        super().__init__()
        fixed_per_site = 3 * output_dim
        width_cost = input_dim + output_dim + 1
        hidden = max(
            1,
            round((target_parameters / sites - fixed_per_site) / width_cost),
        )
        self.sites = sites
        self.networks = nn.ModuleList(
            nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, output_dim),
                nn.LayerNorm(output_dim),
            )
            for _ in range(sites)
        )

    def forward(self, value: Tensor) -> Tensor:
        if value.shape[-2] != self.sites:
            raise ValueError(f"sitewise MLP expects {self.sites} sites")
        return torch.stack(
            [network(value[..., site, :]) for site, network in enumerate(self.networks)],
            dim=-2,
        )


class HierarchicalConditionalMLPDecoder(_DecoderControl):
    """Preserve fixed anatomical routing while replacing both MPS operators."""

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
        **_: object,
    ) -> None:
        super().__init__(
            spec,
            offsets,
            physical_dim=physical_dim,
            frame_width=frame_width,
            groups=groups,
            dropout=dropout,
        )
        self.plan_dim = plan_dim
        self.part_identity = nn.Embedding(6, plan_dim)
        self.joint_identity = nn.Embedding(spec.joint_count, identity_dim)
        self.register_buffer(
            "part_for_joint", torch.tensor(anatomical_part_indices(spec), dtype=torch.long)
        )
        part_condition_dim = 2 * plan_dim
        joint_condition_dim = realization_dim + physical_dim + identity_dim
        self.part_operator = _SitewiseConditionalMLP(
            6,
            part_condition_dim,
            physical_dim,
            target_parameters=_mps_parameter_count(
                6, physical_dim, part_bond_dim, part_condition_dim
            ),
        )
        self.joint_operator = _SitewiseConditionalMLP(
            spec.joint_count,
            joint_condition_dim,
            physical_dim,
            target_parameters=_mps_parameter_count(
                spec.joint_count,
                physical_dim,
                joint_bond_dim,
                joint_condition_dim,
            ),
        )

    def forward(
        self, plan_embeddings: Tensor, plan_lengths: Tensor, realization: NativeRealization
    ) -> DecodedMotion:
        values = realization.values
        plan = upsample_plan(
            plan_embeddings, plan_lengths, realization.lengths, frames=values.shape[1]
        )
        part_identity = self.part_identity.weight[None, None].expand(
            values.shape[0], values.shape[1], -1, -1
        )
        part_input = torch.cat((plan.unsqueeze(2).expand(-1, -1, 6, -1), part_identity), dim=-1)
        part_states = self.part_operator(part_input)
        routed = part_states[:, :, self.part_for_joint]
        joint_identity = self.joint_identity.weight[None, None].expand(
            values.shape[0], values.shape[1], -1, -1
        )
        joint_input = torch.cat(
            (
                values.unsqueeze(2).expand(-1, -1, self.spec.joint_count, -1),
                routed,
                joint_identity,
            ),
            dim=-1,
        )
        return self._motion(self.joint_operator(joint_input), realization.lengths)


class RandomRoutingDecoder(AnatomicalDecoder):
    def __init__(self, spec: DatasetSpec, offsets: Tensor, **kwargs: Any) -> None:
        super().__init__(spec, offsets, **kwargs)
        anatomical = self.part_for_joint.clone()
        counts = torch.bincount(anatomical, minlength=6)
        labels = torch.repeat_interleave(torch.arange(6), counts)
        permutation = torch.randperm(
            spec.joint_count, generator=torch.Generator().manual_seed(2027)
        )
        randomized = torch.empty_like(labels)
        randomized[permutation] = labels
        self.part_for_joint.copy_(randomized)


def build_decoder_control(
    variant: str,
    spec: DatasetSpec,
    offsets: Tensor,
    **kwargs: object,
) -> nn.Module:
    builders: dict[str, type[nn.Module]] = {
        "anatomical": AnatomicalDecoder,
        "shared_decoder": SharedContextDecoder,
        "flat_mps": FlatMPSDecoder,
        "hierarchical_conditional_mlp": HierarchicalConditionalMLPDecoder,
        "random_routing_seed2027": RandomRoutingDecoder,
    }
    try:
        builder = builders[variant]
    except KeyError as error:
        raise KeyError(f"unknown decoder control: {variant}") from error
    return builder(spec, offsets, **kwargs)


class DeterministicRealizationRegressor(nn.Module):
    """Predict one normalized realization without noise or ODE integration."""

    def __init__(
        self,
        *,
        plan_dim: int = 256,
        width: int = 384,
        blocks: int = 8,
        heads: int = 6,
        ffn: int = 1536,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.network = ConditionalFlow(
            plan_dim=plan_dim,
            width=width,
            blocks=blocks,
            heads=heads,
            ffn=ffn,
            dropout=dropout,
        )

    def forward(
        self,
        text: TextFeatures,
        plan_embeddings: Tensor,
        plan_lengths: Tensor,
        lengths: Tensor,
    ) -> NormalizedRealization:
        frames = int(lengths.max().item())
        state = plan_embeddings.new_zeros(lengths.shape[0], frames, 128)
        time = state.new_zeros(lengths.shape[0])
        values = self.network(
            state,
            time,
            text,
            plan_embeddings,
            plan_lengths,
            lengths,
        )
        return NormalizedRealization(values, lengths)

    @staticmethod
    def loss(prediction: NormalizedRealization, target: NormalizedRealization) -> Tensor:
        if not torch.equal(prediction.lengths, target.lengths):
            raise ValueError("prediction and target lengths differ")
        mask = lengths_to_mask(target.lengths, target.values.shape[1])
        squared = (prediction.values - target.values).square() * mask.unsqueeze(-1)
        per_sequence = squared.sum(dim=(1, 2)) / (target.lengths * target.values.shape[-1])
        return per_sequence.mean()
