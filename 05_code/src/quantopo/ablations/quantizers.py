"""Drop-in EMA-VQ, SimVQ, and finite-scalar plan quantizers."""

from __future__ import annotations

import itertools
from typing import cast

import torch
from torch import Tensor, nn

from quantopo.representation.quantizer import EMAQuantizer, QuantizedPlan


def _validate(hidden: Tensor, mask: Tensor, embedding_dim: int) -> None:
    if hidden.ndim != 3 or hidden.shape[-1] != embedding_dim:
        raise ValueError(f"hidden must have shape [B,L,{embedding_dim}]")
    if mask.shape != hidden.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("mask must be boolean with shape [B,L]")


def _utilization(indices: Tensor, mask: Tensor, codebook_size: int) -> float:
    used = int(torch.unique(indices[mask]).numel())
    return used / codebook_size


class SimVQ(nn.Module):
    def __init__(self, codebook_size: int = 512, embedding_dim: int = 256) -> None:
        super().__init__()
        if codebook_size != 512:
            raise ValueError("SimVQ requires 512 codes")
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        generator = torch.Generator().manual_seed(1701)
        basis = torch.randn(codebook_size, embedding_dim, generator=generator)
        basis = basis / torch.linalg.vector_norm(basis, dim=-1, keepdim=True).clamp_min(1e-6)
        self.register_buffer("basis", basis)
        self.transform = nn.Linear(embedding_dim, embedding_dim, bias=False)
        with torch.no_grad():
            self.transform.weight.copy_(torch.eye(embedding_dim))

    @property
    def embedding(self) -> Tensor:
        return cast(Tensor, self.transform(self.basis))

    def freeze(self) -> None:
        self.requires_grad_(False)

    def unfreeze(self) -> None:
        self.requires_grad_(True)

    def forward(
        self, hidden: Tensor, mask: Tensor, *, generator: torch.Generator | None = None
    ) -> QuantizedPlan:
        del generator
        _validate(hidden, mask, self.embedding_dim)
        codebook = self.embedding
        distance = (
            hidden.square().sum(dim=-1, keepdim=True)
            - 2.0 * hidden @ codebook.transpose(0, 1)
            + codebook.square().sum(dim=-1)
        )
        indices = distance.argmin(dim=-1).masked_fill(~mask, 0)
        selected = codebook[indices]
        valid = mask.unsqueeze(-1).to(hidden.dtype)
        embeddings = (hidden + (selected - hidden).detach()) * valid
        commitment = (hidden[mask] - selected[mask].detach()).square().mean()
        codebook_loss = (selected[mask] - hidden[mask].detach()).square().mean()
        return QuantizedPlan(
            indices,
            embeddings,
            mask,
            commitment + codebook_loss,
            hidden.new_tensor(_utilization(indices, mask, self.codebook_size)),
        )


class FiniteScalarQuantizer(nn.Module):
    def __init__(self, codebook_size: int = 512, embedding_dim: int = 256) -> None:
        super().__init__()
        if codebook_size != 8**3:
            raise ValueError("FSQ requires three eight-level scalars")
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.encoder = nn.Linear(embedding_dim, 3)
        self.decoder = nn.Linear(3, embedding_dim)
        levels = torch.tensor(list(itertools.product(range(8), repeat=3)), dtype=torch.float32)
        self.register_buffer("levels", levels)

    @property
    def embedding(self) -> Tensor:
        normalized = self.levels / 3.5 - 1.0
        return cast(Tensor, self.decoder(normalized))

    def freeze(self) -> None:
        self.requires_grad_(False)

    def unfreeze(self) -> None:
        self.requires_grad_(True)

    def forward(
        self, hidden: Tensor, mask: Tensor, *, generator: torch.Generator | None = None
    ) -> QuantizedPlan:
        del generator
        _validate(hidden, mask, self.embedding_dim)
        bounded = (torch.tanh(self.encoder(hidden)) + 1.0) * 3.5
        rounded = bounded + (bounded.round().clamp(0, 7) - bounded).detach()
        digits = rounded.detach().to(torch.long)
        indices = (digits[..., 0] * 64 + digits[..., 1] * 8 + digits[..., 2]).masked_fill(~mask, 0)
        normalized = rounded / 3.5 - 1.0
        decoded = self.decoder(normalized)
        embeddings = decoded * mask.unsqueeze(-1).to(decoded.dtype)
        commitment = (hidden[mask] - decoded[mask].detach()).square().mean()
        return QuantizedPlan(
            indices,
            embeddings,
            mask,
            commitment,
            hidden.new_tensor(_utilization(indices, mask, self.codebook_size)),
        )


def build_quantizer(name: str, *, codebook_size: int = 512, embedding_dim: int = 256) -> nn.Module:
    if name == "ema_vq":
        return EMAQuantizer(codebook_size, embedding_dim)
    if name == "sim_vq":
        return SimVQ(codebook_size, embedding_dim)
    if name == "fsq":
        return FiniteScalarQuantizer(codebook_size, embedding_dim)
    raise KeyError(f"unknown quantizer control: {name}")
