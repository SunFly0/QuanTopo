"""Conditional matrix-product-state feature operator."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn


def _normalize_environment(value: Tensor) -> Tensor:
    return cast(
        Tensor,
        value / torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(1e-6),
    )


class ConditionalMPS(nn.Module):
    def __init__(self, sites: int, physical_dim: int, bond_dim: int, condition_dim: int) -> None:
        super().__init__()
        if sites < 1 or physical_dim < 1 or bond_dim < 1 or condition_dim < 1:
            raise ValueError("MPS dimensions must be positive")
        self.sites = sites
        self.physical_dim = physical_dim
        self.bond_dim = bond_dim
        self.condition_dim = condition_dim
        bonds = [1] + [bond_dim] * (sites - 1) + [1]
        self.cores = nn.ParameterList(
            nn.Parameter(
                torch.randn(bonds[site], physical_dim, bonds[site + 1]) / math.sqrt(physical_dim)
            )
            for site in range(sites)
        )
        self.probes = nn.Parameter(torch.randn(sites, physical_dim))
        self.conditioners = nn.ModuleList(
            nn.Linear(condition_dim, 2 * physical_dim) for _ in range(sites)
        )
        self.normalizers = nn.ModuleList(nn.LayerNorm(physical_dim) for _ in range(sites))

    def forward(self, condition: Tensor) -> Tensor:
        if condition.shape[-2:] != (self.sites, self.condition_dim):
            raise ValueError(
                f"condition must end in [{self.sites},{self.condition_dim}], got {condition.shape}"
            )
        output_dtype = condition.dtype
        leading = condition.shape[:-2]
        flat = condition.reshape(-1, self.sites, self.condition_dim)
        modulations = [layer(flat[:, site]).float() for site, layer in enumerate(self.conditioners)]
        device_type = condition.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            transfers: list[Tensor] = []
            for site in range(self.sites):
                gamma, beta = modulations[site].chunk(2, dim=-1)
                probe = _normalize_environment(self.probes[site].float())
                core = self.cores[site].float()
                transfer = torch.einsum("akc,nk,k->nac", core, 1.0 + gamma, probe)
                transfer = transfer + (beta * probe).sum(dim=-1)[:, None, None]
                transfers.append(transfer)

            left: list[Tensor] = [flat.new_ones(flat.shape[0], 1, dtype=torch.float32)]
            for site in range(self.sites - 1):
                left.append(
                    _normalize_environment(torch.einsum("na,nac->nc", left[-1], transfers[site]))
                )
            right: list[Tensor] = [flat.new_empty(0, dtype=torch.float32)] * self.sites
            right[-1] = flat.new_ones(flat.shape[0], 1, dtype=torch.float32)
            for site in range(self.sites - 1, 0, -1):
                right[site - 1] = _normalize_environment(
                    torch.einsum("nac,nc->na", transfers[site], right[site])
                )

            outputs: list[Tensor] = []
            for site in range(self.sites):
                gamma, beta = modulations[site].chunk(2, dim=-1)
                physical = torch.einsum(
                    "na,akc,nc->nk", left[site], self.cores[site].float(), right[site]
                )
                physical = physical * (1.0 + gamma)
                physical = physical + beta * left[site].sum(dim=-1, keepdim=True) * right[site].sum(
                    dim=-1, keepdim=True
                )
                outputs.append(self.normalizers[site](physical))
            result = torch.stack(outputs, dim=-2)
        return result.reshape(*leading, self.sites, self.physical_dim).to(output_dtype)
