"""Fitted target artifacts used by trajectory-summary ablations."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor, nn


class SummaryTarget(Protocol):
    def fit(self, states: Tensor, mask: Tensor, reference_descriptors: Tensor) -> None: ...
    def forward(self, states: Tensor, mask: Tensor) -> Tensor: ...


@dataclass(frozen=True)
class ArtifactBinding:
    source_digest: str
    variant_fingerprint: str


class FrozenBlockScales(nn.Module):
    """Training-fitted block-distance scales, never estimated per evaluation motion."""

    def __init__(self, blocks: tuple[tuple[int, int], ...]) -> None:
        super().__init__()
        if not blocks or any(start < 0 or end <= start for start, end in blocks):
            raise ValueError("block bounds must be non-empty increasing intervals")
        self.blocks = blocks
        self.register_buffer("scales", torch.ones(len(blocks)))
        self.register_buffer("fitted_flag", torch.tensor(False))

    @torch.no_grad()
    def fit(self, states: Tensor, mask: Tensor) -> None:
        if states.ndim != 3 or mask.shape != states.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("states/mask must have shapes [B,N,C] and [B,N]")
        contributions = []
        for start, end in self.blocks:
            if end > states.shape[-1]:
                raise ValueError("block bounds exceed the state width")
            per_motion = []
            for batch_index in range(states.shape[0]):
                valid = states[batch_index, mask[batch_index], start:end].to(torch.float32)
                if valid.shape[0] < 2:
                    continue
                per_motion.append(torch.pdist(valid, p=2).square().mean())
            if not per_motion:
                raise ValueError("fitting block scales requires two valid states per motion")
            contributions.append(torch.stack(per_motion))
        matrix = torch.stack(contributions, dim=1).to(device="cpu", dtype=torch.float64)
        if bool(torch.any(matrix.sum(dim=0) <= 0)) or bool(torch.any(matrix.sum(dim=1) <= 0)):
            raise ValueError("each training motion and state block needs nonzero pairwise distance")

        weights = torch.ones(matrix.shape[1], dtype=torch.float64)
        target = torch.full_like(weights, 1.0 / matrix.shape[1])
        for _ in range(1_000):
            weighted = matrix * weights
            mean_share = (weighted / weighted.sum(dim=1, keepdim=True)).mean(dim=0)
            if bool(torch.max(torch.abs(mean_share - target)) <= 1e-10):
                break
            weights *= target / mean_share.clamp_min(1e-12)
            weights /= torch.exp(torch.log(weights).mean())
        else:
            raise RuntimeError("block-scale fitting did not converge")

        self.scales.copy_(weights.rsqrt().to(self.scales))
        self.fitted_flag.fill_(True)

    def forward(self, states: Tensor) -> Tensor:
        if not bool(self.fitted_flag):
            raise RuntimeError("fit block scales on the training split before use")
        result = states.clone()
        for index, (start, end) in enumerate(self.blocks):
            result[..., start:end] /= self.scales[index].to(result)
        return result


def save_fitted_artifact(
    path: str | Path,
    module: nn.Module,
    binding: ArtifactBinding,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"format_version": 1, "binding": asdict(binding), "state_dict": module.state_dict()},
        destination,
    )


def load_fitted_artifact(
    path: str | Path,
    module: nn.Module,
    expected: ArtifactBinding,
) -> str:
    source = Path(path)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if payload.get("format_version") != 1 or payload.get("binding") != asdict(expected):
        raise ValueError("fitted ablation artifact provenance does not match")
    module.load_state_dict(payload["state_dict"], strict=True)
    return hashlib.sha256(source.read_bytes()).hexdigest()
