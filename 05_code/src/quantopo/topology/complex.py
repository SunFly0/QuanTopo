"""Sparse time-augmented flag filtrations with a deterministic terminal cone."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor


@dataclass(frozen=True)
class Simplex:
    value: float
    vertices: tuple[int, ...]

    @property
    def dimension(self) -> int:
        return len(self.vertices) - 1


@dataclass(frozen=True)
class Filtration:
    simplices: tuple[Simplex, ...]

    def __post_init__(self) -> None:
        ordered = tuple(
            sorted(self.simplices, key=lambda item: (item.value, item.dimension, item.vertices))
        )
        if ordered != self.simplices:
            object.__setattr__(self, "simplices", ordered)
        vertices = {simplex.vertices for simplex in self.simplices}
        if len(vertices) != len(self.simplices):
            raise ValueError("filtration contains duplicate simplices")


def _pairwise_distance(states: Tensor, time_weight: float) -> Tensor:
    count, width = states.shape
    time = torch.linspace(0.0, 1.0, count, dtype=states.dtype, device=states.device)
    state_sq = (states[:, None] - states[None, :]).square().sum(dim=-1) / width
    time_sq = time_weight**2 * (time[:, None] - time[None, :]).abs().square()
    return torch.sqrt(state_sq + time_sq + 1e-12)


def build_flag_filtration(
    states: Tensor,
    knn_k: int = 6,
    time_weight: float = 0.15,
    *,
    terminal_cone: bool = True,
) -> Filtration:
    """Build the symmetrized kNN/temporal flag complex through dimension two."""

    if states.ndim != 2 or states.shape[0] < 1:
        raise ValueError("states must have shape [N,D] with N >= 1")
    if states.shape[0] > 64:
        raise ValueError("the persistence implementation supports at most 64 states")
    if knn_k < 0 or time_weight < 0:
        raise ValueError("knn_k and time_weight must be nonnegative")
    count = states.shape[0]
    distance = _pairwise_distance(states.to(torch.float32), time_weight).cpu()
    edges: dict[tuple[int, int], float] = {}
    degree = min(knn_k, count - 1)
    for source in range(count):
        targets = sorted(
            (target for target in range(count) if target != source),
            key=lambda target: (float(distance[source, target]), target),
        )[:degree]
        for target in targets:
            edge = (min(source, target), max(source, target))
            edges[edge] = float(distance[edge])
    for source in range(count - 1):
        edge = (source, source + 1)
        edges[edge] = float(distance[edge])

    if terminal_cone:
        maximum_original = max(edges.values(), default=0.0)
        for target in range(1, count):
            edges.setdefault((0, target), maximum_original)

    simplices: list[Simplex] = [Simplex(0.0, (vertex,)) for vertex in range(count)]
    simplices.extend(Simplex(value, edge) for edge, value in edges.items())
    for triangle in itertools.combinations(range(count), 3):
        triangle_edges = cast(
            tuple[tuple[int, int], ...], tuple(itertools.combinations(triangle, 2))
        )
        if all(edge in edges for edge in triangle_edges):
            simplices.append(Simplex(max(edges[edge] for edge in triangle_edges), triangle))
    return Filtration(
        tuple(sorted(simplices, key=lambda item: (item.value, item.dimension, item.vertices)))
    )
