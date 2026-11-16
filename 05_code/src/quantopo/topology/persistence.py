"""Deterministic persistence reduction over the field F2."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .complex import Filtration, build_flag_filtration
from .states import normalize_states, normalize_states_block_balanced


@dataclass(frozen=True)
class PersistenceDiagram:
    h0: tuple[tuple[float, float], ...]
    h1: tuple[tuple[float, float], ...]
    h0_order: tuple[int, ...] = ()
    h1_order: tuple[int, ...] = ()


@dataclass(frozen=True)
class FixedPersistence:
    points: Tensor
    mask: Tensor
    raw: PersistenceDiagram


def _boundary(simplex: tuple[int, ...], lookup: dict[tuple[int, ...], int]) -> int:
    if len(simplex) == 1:
        return 0
    column = 0
    for omitted in range(len(simplex)):
        face = simplex[:omitted] + simplex[omitted + 1 :]
        column ^= 1 << lookup[face]
    return column


def reduce_f2(filtration: Filtration) -> PersistenceDiagram:
    """Reduce boundary columns as Python bitsets with deterministic pivot pairing."""

    simplices = filtration.simplices
    lookup = {simplex.vertices: index for index, simplex in enumerate(simplices)}
    reduced: list[int] = [0] * len(simplices)
    pivot_column: dict[int, int] = {}
    pairs: dict[int, list[tuple[float, float, int]]] = {0: [], 1: []}
    for column_index, simplex in enumerate(simplices):
        column = _boundary(simplex.vertices, lookup)
        while column:
            pivot = column.bit_length() - 1
            previous = pivot_column.get(pivot)
            if previous is None:
                break
            column ^= reduced[previous]
        reduced[column_index] = column
        if not column:
            continue
        pivot = column.bit_length() - 1
        pivot_column[pivot] = column_index
        homology_dimension = simplices[pivot].dimension
        if homology_dimension in pairs and simplex.value > simplices[pivot].value:
            pairs[homology_dimension].append((simplices[pivot].value, simplex.value, column_index))

    def ordered(dimension: int) -> tuple[tuple[tuple[float, float], ...], tuple[int, ...]]:
        values = sorted(pairs[dimension], key=lambda item: (item[0], item[1], item[2]))
        return tuple((birth, death) for birth, death, _ in values), tuple(
            order for _, _, order in values
        )

    h0, h0_order = ordered(0)
    h1, h1_order = ordered(1)
    return PersistenceDiagram(h0, h1, h0_order, h1_order)


def compute_persistence(
    states: Tensor,
    *,
    knn_k: int = 6,
    time_weight: float = 0.15,
    point_slots: int = 16,
    lifetime_threshold: float = 1e-5,
    terminal_cone: bool = True,
    block_balanced: bool = False,
    block_scales: Tensor | None = None,
) -> FixedPersistence:
    """Normalize states, reduce their filtration, and retain fixed-size H0/H1 points."""

    states = states.to(torch.float32)
    if block_balanced:
        if block_scales is None:
            raise RuntimeError("block-balanced persistence requires fitted training block scales")
        normalized = normalize_states_block_balanced(states, block_scales)
    else:
        if block_scales is not None:
            raise ValueError("block_scales are valid only for block-balanced persistence")
        normalized = normalize_states(states)
    raw = reduce_f2(
        build_flag_filtration(
            normalized,
            knn_k=knn_k,
            time_weight=time_weight,
            terminal_cone=terminal_cone,
        )
    )
    points = torch.zeros(2, point_slots, 2, dtype=torch.float32, device=states.device)
    mask = torch.zeros(2, point_slots, dtype=torch.bool, device=states.device)
    for dimension, (pairs, orders) in enumerate(((raw.h0, raw.h0_order), (raw.h1, raw.h1_order))):
        candidates = [
            (birth, death, order)
            for (birth, death), order in zip(pairs, orders, strict=True)
            if death - birth >= lifetime_threshold
        ]
        candidates.sort(key=lambda item: (-(item[1] - item[0]), item[0], item[1], item[2]))
        for slot, (birth, death, _) in enumerate(candidates[:point_slots]):
            points[dimension, slot] = torch.tensor(
                [birth, death], dtype=points.dtype, device=points.device
            )
            mask[dimension, slot] = True
    return FixedPersistence(points, mask, raw)
