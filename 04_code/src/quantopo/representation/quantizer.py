"""Synchronized exponential-moving-average vector quantization."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as distributed
from torch import Tensor, nn


@dataclass(frozen=True)
class QuantizedPlan:
    indices: Tensor
    embeddings: Tensor
    mask: Tensor
    commitment_loss: Tensor
    utilization: Tensor


def masked_sequence_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Average valid values within each sequence, then weight sequences equally."""

    if values.shape != mask.shape or mask.dtype != torch.bool or values.ndim != 2:
        raise ValueError("values and mask must have matching [B,L] shapes")
    counts = mask.sum(dim=1)
    if bool(torch.any(counts == 0)):
        raise ValueError("each sequence must contain at least one valid value")
    per_sequence = (values * mask.to(values.dtype)).sum(dim=1) / counts.to(values.dtype)
    return per_sequence.mean()


class EMAQuantizer(nn.Module):
    def __init__(
        self,
        codebook_size: int,
        embedding_dim: int,
        decay: float = 0.99,
        smoothing: float = 1e-5,
        dead_code_start: int = 100,
        dead_code_interval: int = 100,
        dead_code_threshold: float = 1.0,
    ) -> None:
        super().__init__()
        if codebook_size < 2 or embedding_dim < 1 or not 0 < decay < 1:
            raise ValueError("invalid EMA quantizer dimensions or decay")
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.smoothing = smoothing
        self.dead_code_start = dead_code_start
        self.dead_code_interval = dead_code_interval
        self.dead_code_threshold = dead_code_threshold
        self.register_buffer("embedding", torch.zeros(codebook_size, embedding_dim))
        self.register_buffer("ema_count", torch.zeros(codebook_size))
        self.register_buffer("ema_sum", torch.zeros(codebook_size, embedding_dim))
        self.register_buffer("initialized", torch.tensor(False))
        self.register_buffer("update_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("updates_enabled", torch.tensor(True))

    def freeze(self) -> None:
        self.updates_enabled.fill_(False)

    def unfreeze(self) -> None:
        self.updates_enabled.fill_(True)

    def _global_features(self, features: Tensor) -> Tensor:
        if not distributed.is_available() or not distributed.is_initialized():
            return features
        gathered: list[Tensor | None] = [None] * distributed.get_world_size()
        distributed.all_gather_object(gathered, features.detach().cpu())
        merged = torch.cat([item for item in gathered if item is not None], dim=0)
        return merged.to(features.device, features.dtype)

    @torch.no_grad()
    def _initialize(self, features: Tensor, generator: torch.Generator | None) -> None:
        features = self._global_features(features)
        if features.shape[0] < 1:
            raise ValueError("cannot initialize codebook without valid features")
        if generator is None:
            selection = torch.arange(self.codebook_size, device=features.device) % features.shape[0]
        else:
            selection = torch.randint(
                features.shape[0],
                (self.codebook_size,),
                generator=generator,
                device=features.device,
            )
        initial = features[selection]
        self.embedding.copy_(initial)
        self.ema_sum.copy_(initial)
        self.ema_count.fill_(1.0)
        self.initialized.fill_(True)

    @torch.no_grad()
    def _ema_update(self, features: Tensor, indices: Tensor) -> None:
        counts = torch.bincount(indices, minlength=self.codebook_size).to(features.dtype)
        sums = torch.zeros_like(self.ema_sum)
        sums.index_add_(0, indices, features)
        if distributed.is_available() and distributed.is_initialized():
            distributed.all_reduce(counts)
            distributed.all_reduce(sums)
        self.ema_count.mul_(self.decay).add_(counts, alpha=1.0 - self.decay)
        self.ema_sum.mul_(self.decay).add_(sums, alpha=1.0 - self.decay)
        total = self.ema_count.sum()
        smoothed = (
            (self.ema_count + self.smoothing)
            / (total + self.codebook_size * self.smoothing)
            * total
        )
        self.embedding.copy_(self.ema_sum / smoothed.unsqueeze(-1).clamp_min(self.smoothing))

    @torch.no_grad()
    def _reset_dead(self, features: Tensor, generator: torch.Generator | None) -> None:
        dead = torch.nonzero(self.ema_count < self.dead_code_threshold).flatten()
        if not dead.numel():
            return
        global_features = self._global_features(features)
        replacements = global_features.new_empty((dead.numel(), self.embedding_dim))
        using_ddp = distributed.is_available() and distributed.is_initialized()
        if not using_ddp or distributed.get_rank() == 0:
            if generator is None:
                selection = (
                    torch.arange(dead.numel(), device=global_features.device)
                    % global_features.shape[0]
                )
            else:
                selection = torch.randint(
                    global_features.shape[0],
                    (dead.numel(),),
                    generator=generator,
                    device=global_features.device,
                )
            replacements.copy_(global_features[selection])
        if using_ddp:
            distributed.broadcast(replacements, src=0)
        self.embedding[dead] = replacements
        self.ema_sum[dead] = replacements
        self.ema_count[dead] = 1.0

    def forward(
        self,
        hidden: Tensor,
        mask: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> QuantizedPlan:
        if hidden.ndim != 3 or hidden.shape[-1] != self.embedding_dim:
            raise ValueError(f"hidden must have shape [B,L,{self.embedding_dim}]")
        if mask.shape != hidden.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("mask must have shape [B,L]")
        valid_features = hidden[mask]
        if valid_features.shape[0] < 1:
            raise ValueError("quantization requires at least one valid feature")
        updating = self.training and bool(self.updates_enabled.item())
        if updating and not bool(self.initialized.item()):
            self._initialize(valid_features.detach(), generator)
        elif not bool(self.initialized.item()):
            raise RuntimeError("EMA codebook must be initialized before frozen inference")

        distance = (
            hidden.square().sum(dim=-1, keepdim=True)
            - 2.0 * hidden @ self.embedding.transpose(0, 1)
            + self.embedding.square().sum(dim=-1)
        )
        indices = distance.argmin(dim=-1)
        valid_indices = indices[mask]
        selected = self.embedding[indices]
        commitment_error = (hidden - selected.detach()).square().mean(dim=-1)
        commitment = masked_sequence_mean(commitment_error, mask)
        straight_through = hidden + (selected - hidden).detach()
        straight_through = straight_through * mask.unsqueeze(-1).to(hidden.dtype)
        indices = indices.masked_fill(~mask, 0)

        if updating:
            self._ema_update(valid_features.detach(), valid_indices.detach())
            self.update_count.add_(1)
            step = int(self.update_count.item())
            if step >= self.dead_code_start and step % self.dead_code_interval == 0:
                self._reset_dead(valid_features.detach(), generator)
        utilization = valid_indices.unique().numel() / self.codebook_size
        return QuantizedPlan(
            indices,
            straight_through,
            mask,
            commitment,
            hidden.new_tensor(utilization),
        )
