"""Matched trajectory summaries, prediction heads, and topology sensitivities."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import asdict, dataclass, replace
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from quantopo.contracts.config import TopologyConfig
from quantopo.text.topology_predictor import PredictedPersistence
from quantopo.topology.persistence import (
    FixedPersistence,
    PersistenceDiagram,
    compute_persistence,
)
from quantopo.topology.readout import PersistenceReadout


def _validate(states: Tensor, mask: Tensor) -> None:
    if states.ndim != 3 or mask.shape != states.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("states/mask must have shapes [B,N,C] and boolean [B,N]")
    if not bool(mask.any(dim=1).all()):
        raise ValueError("each summary input needs at least one valid state")


class CenteredPCA(nn.Module):
    """Train-fitted centered PCA with deterministic signs and PH variance matching."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.register_buffer("mean", torch.empty(0))
        self.register_buffer("components", torch.empty(0, output_dim))
        self.register_buffer("variance_scale", torch.tensor(1.0))
        self.register_buffer("fitted_flag", torch.tensor(False))

    @property
    def fitted(self) -> bool:
        return bool(self.fitted_flag.item())

    @torch.no_grad()
    def fit(self, features: Tensor, reference_descriptors: Tensor) -> None:
        if features.ndim != 2 or features.shape[0] < 2:
            raise ValueError("PCA fitting requires at least two training rows")
        if reference_descriptors.shape != (features.shape[0], self.output_dim):
            raise ValueError("reference descriptors must have shape [B,output_dim]")
        values = features.detach().to(device="cpu", dtype=torch.float64)
        reference = reference_descriptors.detach().to(device="cpu", dtype=torch.float64)
        mean = values.mean(dim=0)
        centered = values - mean
        _, _, right = torch.linalg.svd(centered, full_matrices=False)
        rank = min(right.shape[0], self.output_dim)
        components = torch.zeros(values.shape[1], self.output_dim, dtype=torch.float64)
        components[:, :rank] = right[:rank].transpose(0, 1)
        for column in range(rank):
            vector = components[:, column]
            pivot = int(vector.abs().argmax().item())
            if vector[pivot] < 0:
                components[:, column].neg_()
        projected = centered @ components
        projected_variance = projected.var(dim=0, correction=0).sum()
        reference_variance = (reference - reference.mean(dim=0)).var(dim=0, correction=0).sum()
        scale = torch.sqrt(reference_variance / projected_variance.clamp_min(1e-12))
        self.mean = mean.to(torch.float32)
        self.components = components.to(torch.float32)
        self.variance_scale = scale.to(torch.float32)
        self.fitted_flag.fill_(True)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ) -> None:
        # PCA widths are learned during fit; restore their shapes before copying.
        mean = state_dict.get(prefix + "mean")
        components = state_dict.get(prefix + "components")
        if mean is not None and components is not None:
            if (
                mean.ndim != 1 or components.ndim != 2
                or components.shape != (mean.numel(), self.output_dim)
            ):
                error_msgs.append(prefix + "PCA state has incompatible feature/output dimensions")
            else:
                self.mean = self.mean.new_empty(mean.shape)
                self.components = self.components.new_empty(components.shape)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def forward(self, features: Tensor) -> Tensor:
        if not self.fitted:
            raise RuntimeError("fit PCA on training summaries before use")
        if features.ndim != 2 or features.shape[-1] != self.mean.numel():
            raise ValueError("summary features do not match the fitted PCA width")
        centered = features - self.mean.to(features)
        return (centered @ self.components.to(features)) * self.variance_scale.to(features)


class ProjectedSummary(nn.Module):
    def __init__(self, descriptor_dim: int) -> None:
        super().__init__()
        self.pca = CenteredPCA(descriptor_dim)

    def features(self, states: Tensor, mask: Tensor) -> Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def fit(self, states: Tensor, mask: Tensor, reference_descriptors: Tensor) -> None:
        self.pca.fit(self.features(states, mask), reference_descriptors)

    def forward(self, states: Tensor, mask: Tensor) -> Tensor:
        return cast(Tensor, self.pca(self.features(states, mask)))


class NoSummary(nn.Module):
    def __init__(self, descriptor_dim: int) -> None:
        super().__init__()
        self.descriptor_dim = descriptor_dim

    def forward(self, states: Tensor, mask: Tensor) -> Tensor:
        _validate(states, mask)
        zero = states.sum(dim=(1, 2), keepdim=False).unsqueeze(-1) * 0.0
        return zero.expand(-1, self.descriptor_dim)


class AutocorrelationSummary(ProjectedSummary):
    def __init__(self, descriptor_dim: int, max_lag: int = 16) -> None:
        super().__init__(descriptor_dim)
        self.max_lag = max_lag

    def features(self, states: Tensor, mask: Tensor) -> Tensor:
        _validate(states, mask)
        weight = mask.unsqueeze(-1).to(states.dtype)
        second_moment = (states.square() * weight).sum(dim=1)
        second_moment = second_moment / weight.sum(dim=1).clamp_min(1.0)
        values: list[Tensor] = []
        valid_lags: list[Tensor] = []
        for lag in range(1, self.max_lag + 1):
            if lag >= states.shape[1]:
                pair_mask = mask.new_zeros(mask.shape[0], 0)
                correlation = states.new_zeros(states.shape[0], states.shape[2])
            else:
                pair_mask = mask[:, :-lag] & mask[:, lag:]
                pair_weight = pair_mask.unsqueeze(-1).to(states.dtype)
                correlation = (states[:, :-lag] * states[:, lag:] * pair_weight).sum(dim=1)
                correlation = correlation / pair_weight.sum(dim=1).clamp_min(1.0)
                correlation = correlation / second_moment.clamp_min(1e-6)
            values.append(correlation)
            valid_lags.append(pair_mask.any(dim=1, keepdim=True).to(states.dtype))
        return torch.cat((*values, *valid_lags), dim=-1)


class PowerSpectrumSummary(ProjectedSummary):
    def __init__(self, descriptor_dim: int, frequencies: int = 16) -> None:
        super().__init__(descriptor_dim)
        self.frequencies = frequencies

    def features(self, states: Tensor, mask: Tensor) -> Tensor:
        _validate(states, mask)
        rows: list[Tensor] = []
        for batch_index in range(states.shape[0]):
            valid = states[batch_index, mask[batch_index]]
            # Evaluate the same normalized frequency grid for every sequence.
            time = torch.arange(valid.shape[0], device=valid.device, dtype=valid.dtype)
            frequencies = torch.arange(
                1, self.frequencies + 1, device=valid.device, dtype=valid.dtype
            ) / (2.0 * self.frequencies)
            phase = 2.0 * torch.pi * frequencies[:, None] * time[None, :]
            real = torch.cos(phase) @ valid
            imag = -torch.sin(phase) @ valid
            power = (real.square() + imag.square()) / max(valid.shape[0], 1)
            rows.append(power.log1p().flatten())
        return torch.stack(rows)


class DistanceMatrixSummary(ProjectedSummary):
    def __init__(self, descriptor_dim: int, grid: int = 32, time_weight: float = 0.15) -> None:
        super().__init__(descriptor_dim)
        self.grid = grid
        self.time_weight = time_weight
        self.register_buffer("upper", torch.triu_indices(grid, grid, offset=1))

    def features(self, states: Tensor, mask: Tensor) -> Tensor:
        _validate(states, mask)
        rows: list[Tensor] = []
        for batch_index in range(states.shape[0]):
            valid = states[batch_index, mask[batch_index]]
            count = valid.shape[0]
            time = torch.linspace(0.0, 1.0, count, dtype=valid.dtype, device=valid.device)
            state_sq = (valid[:, None] - valid[None, :]).square().sum(dim=-1)
            state_sq = state_sq / valid.shape[-1]
            time_sq = self.time_weight**2 * (time[:, None] - time[None, :]).square()
            distance = torch.sqrt(state_sq + time_sq + 1e-12)
            resized = F.interpolate(
                distance[None, None],
                size=(self.grid, self.grid),
                mode="bilinear",
                align_corners=True,
            )[0, 0]
            rows.append(resized[self.upper[0], self.upper[1]])
        return torch.stack(rows)


class LearnedSetSummary(nn.Module):
    def __init__(self, descriptor_dim: int) -> None:
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.LazyLinear(256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU()
        )
        self.output = nn.Linear(512, descriptor_dim)

    def forward(self, states: Tensor, mask: Tensor) -> Tensor:
        _validate(states, mask)
        features = self.point_mlp(states)
        weight = mask.unsqueeze(-1).to(features.dtype)
        mean = (features * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
        maximum = features.masked_fill(~mask.unsqueeze(-1), float("-inf")).amax(dim=1)
        return cast(Tensor, self.output(torch.cat((mean, maximum), dim=-1)))


class PersistenceSummary(nn.Module):
    """Exact persistence target shared by vector and point-set prediction controls."""

    def __init__(self, descriptor_dim: int) -> None:
        super().__init__()
        if descriptor_dim != 128:
            raise ValueError("persistence summary requires the shared 128-D readout")
        self.readout = PersistenceReadout()

    def forward(self, states: Tensor, mask: Tensor) -> Tensor:
        _validate(states, mask)
        rows: list[Tensor] = []
        for batch_index in range(states.shape[0]):
            valid = states[batch_index, mask[batch_index]]
            diagram = compute_persistence(valid)
            rows.append(self.readout(diagram.points.unsqueeze(0), diagram.mask.unsqueeze(0))[0])
        return torch.stack(rows)


def build_summary(name: str, *, descriptor_dim: int = 128) -> nn.Module:
    builders: dict[str, type[nn.Module]] = {
        "none": NoSummary,
        "acf": AutocorrelationSummary,
        "psd": PowerSpectrumSummary,
        "edm": DistanceMatrixSummary,
        "learned_set": LearnedSetSummary,
        "ph_vector": PersistenceSummary,
        "point_set": PersistenceSummary,
    }
    try:
        return builders[name](descriptor_dim)
    except KeyError as error:
        raise KeyError(f"unknown trajectory summary: {name}") from error


class VectorTopologyHead(nn.Module):
    def __init__(self, *, variant: str = "standard") -> None:
        super().__init__()
        if variant == "standard":
            self.network = nn.Sequential(nn.Linear(1536, 512), nn.SiLU(), nn.Linear(512, 128))
        elif variant == "wide":
            self.network = nn.Sequential(
                nn.Linear(1536, 832),
                nn.SiLU(),
                nn.Linear(832, 384),
                nn.SiLU(),
                nn.Linear(384, 128),
            )
        else:
            raise KeyError(f"unknown vector topology head: {variant}")

    def forward(self, pooled: Tensor) -> Tensor:
        if pooled.shape[-1] != 1536:
            raise ValueError("vector topology head expects 1536 pooled coordinates")
        return cast(Tensor, self.network(pooled))


class _DiagramHead(nn.Module):
    def __init__(self, point_slots: int) -> None:
        super().__init__()
        self.point_slots = point_slots
        self.network = nn.Sequential(
            nn.Linear(1536, 512), nn.SiLU(), nn.Linear(512, point_slots * 3)
        )

    def forward(self, pooled: Tensor) -> tuple[Tensor, Tensor]:
        raw = self.network(pooled).reshape(pooled.shape[0], self.point_slots, 3)
        birth = F.softplus(raw[..., 0])
        death = birth + F.softplus(raw[..., 1]) + 1e-5
        return torch.stack((birth, death), dim=-1), torch.sigmoid(raw[..., 2])


class PointSetPredictionHead(nn.Module):
    def __init__(self, *, readout: PersistenceReadout, point_slots: int = 16) -> None:
        super().__init__()
        self.heads = nn.ModuleList(_DiagramHead(point_slots) for _ in range(2))
        self.readout = readout
        self.readout.requires_grad_(False)

    def forward(self, pooled: Tensor) -> PredictedPersistence:
        results = [head(pooled) for head in self.heads]
        points = torch.stack([result[0] for result in results], dim=1)
        existence = torch.stack([result[1] for result in results], dim=1)
        return PredictedPersistence(points, existence, self.readout(points, existence))


def build_topology_prediction_head(
    name: str,
    *,
    readout: PersistenceReadout | None = None,
    point_slots: int = 16,
) -> nn.Module:
    """Construct the prediction interfaces used in the topology-head comparison."""

    if name == "vector_standard":
        return VectorTopologyHead(variant="standard")
    if name == "vector_wide":
        return VectorTopologyHead(variant="wide")
    if name == "point_set":
        return PointSetPredictionHead(
            readout=PersistenceReadout() if readout is None else readout,
            point_slots=point_slots,
        )
    raise KeyError(f"unknown topology prediction head: {name}")


@dataclass(frozen=True)
class TopologySensitivity:
    name: str
    config: TopologyConfig
    homology: tuple[int, ...]
    terminal_cone: bool
    block_balanced: bool
    cache_fingerprint: str
    block_scale_digest: str | None = None

    @staticmethod
    def _scale_digest(block_scales: Tensor) -> str:
        values = block_scales.detach().to(device="cpu", dtype=torch.float64)
        if values.shape != (3,) or not bool(torch.isfinite(values).all()):
            raise ValueError("block scales must contain three finite values")
        if bool(torch.any(values <= 0)):
            raise ValueError("block scales must be positive")
        return hashlib.sha256(struct.pack("!3d", *values.tolist())).hexdigest()

    def bind_block_scales(self, block_scales: Tensor) -> TopologySensitivity:
        if not self.block_balanced:
            raise ValueError("only the block-balanced variant accepts block scales")
        scale_digest = self._scale_digest(block_scales)
        fingerprint = hashlib.sha256(
            json.dumps(
                {"target": self.cache_fingerprint, "block_scales": scale_digest},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        return replace(
            self,
            cache_fingerprint=fingerprint,
            block_scale_digest=scale_digest,
        )

    def compute(self, states: Tensor, *, block_scales: Tensor | None = None) -> FixedPersistence:
        """Compute this variant while preserving the shared fixed-slot contract."""

        if states.ndim != 2:
            raise ValueError("topology states must have shape [N,C]")
        if self.block_balanced and block_scales is None:
            raise RuntimeError("block-balanced persistence requires fitted training block scales")
        if not self.block_balanced and block_scales is not None:
            raise ValueError("block scales are valid only for the block-balanced variant")
        if self.block_balanced:
            if self.block_scale_digest is None:
                raise RuntimeError("bind fitted training block scales before computing persistence")
            if self._scale_digest(block_scales) != self.block_scale_digest:
                raise ValueError("block scales do not match the bound cache fingerprint")
        count = min(self.config.sample_count, states.shape[0])
        if count < states.shape[0]:
            indices = (
                torch.linspace(0, states.shape[0] - 1, count, device=states.device)
                .round()
                .to(torch.long)
            )
            states = states.index_select(0, indices)

        result = compute_persistence(
            states,
            knn_k=self.config.knn_k,
            time_weight=self.config.time_weight,
            point_slots=self.config.point_slots,
            lifetime_threshold=self.config.lifetime_threshold,
            terminal_cone=self.terminal_cone,
            block_balanced=self.block_balanced,
            block_scales=block_scales,
        )
        points = result.points.clone()
        mask = result.mask.clone()
        for dimension in {0, 1} - set(self.homology):
            points[dimension].zero_()
            mask[dimension].zero_()
        raw = PersistenceDiagram(
            result.raw.h0 if 0 in self.homology else (),
            result.raw.h1 if 1 in self.homology else (),
            result.raw.h0_order if 0 in self.homology else (),
            result.raw.h1_order if 1 in self.homology else (),
        )
        return FixedPersistence(points, mask, raw)


def build_topology_sensitivity(name: str, config: TopologyConfig) -> TopologySensitivity:
    updated = config
    homology: tuple[int, ...] = (0, 1)
    terminal_cone = True
    block_balanced = False
    if name == "h0_only":
        homology = (0,)
    elif name == "h1_only":
        homology = (1,)
    elif name.startswith("states_"):
        updated = replace(config, sample_count=int(name.removeprefix("states_")))
    elif name == "no_time":
        updated = replace(config, time_weight=0.0)
    elif name == "no_terminal_cone":
        terminal_cone = False
    elif name == "block_balanced":
        block_balanced = True
    else:
        raise KeyError(f"unknown topology sensitivity: {name}")
    payload = {
        "config": asdict(updated),
        "homology": homology,
        "terminal_cone": terminal_cone,
        "block_balanced": block_balanced,
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return TopologySensitivity(name, updated, homology, terminal_cone, block_balanced, fingerprint)
