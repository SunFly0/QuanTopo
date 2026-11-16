"""Weighted persistence-diagram matching used in Stage B1."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from quantopo.contracts.tensors import PersistenceBatch
from quantopo.text.topology_predictor import PredictedPersistence
from quantopo.topology.readout import PersistenceReadout


@dataclass(frozen=True)
class SinkhornResult:
    distance: Tensor
    coupling: Tensor
    row_marginal: Tensor
    column_marginal: Tensor


@dataclass(frozen=True)
class DiagramLossTerms:
    total: Tensor
    descriptor: Tensor
    transport: Tensor
    mass: Tensor


def _diagonal_projection(points: Tensor) -> Tensor:
    midpoint = points.mean(dim=-1, keepdim=True)
    return midpoint.expand_as(points)


def augmented_sinkhorn_distance(
    predicted_points: Tensor,
    predicted_mass: Tensor,
    reference_points: Tensor,
    reference_mass: Tensor,
    *,
    epsilon: float = 0.05,
    iterations: int = 30,
    stabilizer: float = 1e-8,
) -> SinkhornResult:
    """Match weighted diagrams after augmenting each by the other's diagonal copies."""

    if predicted_points.ndim != 2 or predicted_points.shape[-1] != 2:
        raise ValueError("predicted_points must have shape [M,2]")
    if reference_points.ndim != 2 or reference_points.shape[-1] != 2:
        raise ValueError("reference_points must have shape [N,2]")
    if predicted_mass.shape != predicted_points.shape[:1]:
        raise ValueError("predicted_mass must have shape [M]")
    if reference_mass.shape != reference_points.shape[:1]:
        raise ValueError("reference_mass must have shape [N]")
    if epsilon <= 0 or iterations < 1 or stabilizer <= 0:
        raise ValueError("epsilon, iterations, and stabilizer must be positive")
    if bool(torch.any(predicted_mass < 0)) or bool(torch.any(reference_mass < 0)):
        raise ValueError("diagram masses must be nonnegative")

    device = predicted_points.device
    original_dtype = predicted_points.dtype
    pred_points = predicted_points.float()
    ref_points = reference_points.to(device=device, dtype=torch.float32)
    pred_mass = predicted_mass.float()
    ref_mass = reference_mass.to(device=device, dtype=torch.float32)
    row_mass = torch.cat((pred_mass, ref_mass), dim=0)
    column_mass = torch.cat((ref_mass, pred_mass), dim=0)
    total_mass = row_mass.sum()
    rows = row_mass.numel()
    columns = column_mass.numel()

    if bool(total_mass.detach() <= 1e-12):
        zero = (pred_points.sum() + pred_mass.sum()) * 0.0
        return SinkhornResult(
            distance=zero.to(original_dtype),
            coupling=torch.zeros(rows, columns, device=device, dtype=torch.float32),
            row_marginal=torch.zeros(rows, device=device, dtype=torch.float32),
            column_marginal=torch.zeros(columns, device=device, dtype=torch.float32),
        )

    row_marginal = row_mass / total_mass
    column_marginal = column_mass / total_mass
    row_points = torch.cat((pred_points, _diagonal_projection(ref_points)), dim=0)
    column_points = torch.cat((ref_points, _diagonal_projection(pred_points)), dim=0)
    cost = torch.cdist(row_points, column_points).square()
    cost[pred_points.shape[0] :, ref_points.shape[0] :] = 0.0

    negative_infinity = torch.tensor(float("-inf"), device=device)
    log_row = torch.where(
        row_marginal > 0,
        row_marginal.clamp_min(torch.finfo(torch.float32).tiny).log(),
        negative_infinity,
    )
    log_column = torch.where(
        column_marginal > 0,
        column_marginal.clamp_min(torch.finfo(torch.float32).tiny).log(),
        negative_infinity,
    )
    kernel = -cost / epsilon
    log_u = torch.zeros_like(row_marginal)
    log_v = torch.zeros_like(column_marginal)
    for _ in range(iterations):
        log_u = log_row - torch.logsumexp(kernel + log_v.unsqueeze(0), dim=1)
        log_v = log_column - torch.logsumexp(kernel + log_u.unsqueeze(1), dim=0)
    coupling = torch.exp(log_u.unsqueeze(1) + kernel + log_v.unsqueeze(0))
    transport_cost = total_mass * (coupling * cost).sum()
    root = torch.sqrt(transport_cost + stabilizer) - stabilizer**0.5
    return SinkhornResult(
        distance=root.to(original_dtype),
        coupling=coupling,
        row_marginal=row_marginal,
        column_marginal=column_marginal,
    )


def diagram_loss(
    prediction: PredictedPersistence,
    reference: PersistenceBatch,
    *,
    readout: PersistenceReadout,
    descriptor_weight: float = 0.5,
    transport_weight: float = 0.1,
    mass_weight: float = 0.05,
) -> DiagramLossTerms:
    """Compute descriptor, augmented transport, and cardinality losses."""

    if prediction.points.shape != reference.points.shape:
        raise ValueError("predicted and reference point tensors must have the same shape")
    if prediction.existence.shape != reference.mask.shape:
        raise ValueError("predicted existence and reference mask must have the same shape")
    if tuple(readout.parameters()):
        raise ValueError("persistence readout must be fixed and parameter-free")

    descriptor = (prediction.descriptor - reference.descriptor.detach()).square().sum(dim=-1)
    descriptor = descriptor.mean() / prediction.descriptor.shape[-1]
    transport_values: list[Tensor] = []
    mass_values: list[Tensor] = []
    point_slots = prediction.points.shape[2]
    for batch_index in range(prediction.points.shape[0]):
        for homology in range(2):
            result = augmented_sinkhorn_distance(
                prediction.points[batch_index, homology],
                prediction.existence[batch_index, homology],
                reference.points[batch_index, homology],
                reference.mask[batch_index, homology].to(prediction.existence.dtype),
            )
            transport_values.append(result.distance)
            difference = (
                prediction.existence[batch_index, homology].sum()
                - reference.mask[batch_index, homology].sum()
            )
            mass_values.append(F.smooth_l1_loss(difference, torch.zeros_like(difference)))

    transport = torch.stack(transport_values).mean()
    mass = torch.stack(mass_values).mean() / point_slots
    total = descriptor_weight * descriptor + transport_weight * transport + mass_weight * mass
    return DiagramLossTerms(total=total, descriptor=descriptor, transport=transport, mass=mass)
