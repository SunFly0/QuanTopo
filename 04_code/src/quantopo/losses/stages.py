"""Loss terms and coefficients for the three training stages."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from quantopo.models.factory import StageALossCoefficients, StageB1LossCoefficients


@dataclass(frozen=True)
class StageALossTerms:
    reconstruction: Tensor
    velocity: Tensor
    commitment: Tensor
    plan_topology: Tensor
    anchor: Tensor


@dataclass(frozen=True)
class StageB1LossTerms:
    diagram: Tensor
    autoregressive: Tensor
    soft_plan: Tensor


@dataclass(frozen=True)
class StageB1Loss:
    diagram: Tensor
    autoregressive: Tensor
    soft_plan: Tensor
    total: Tensor


def combine_stage_a(
    terms: StageALossTerms,
    descriptor_dim: int = 128,
    coefficients: StageALossCoefficients | None = None,
) -> Tensor:
    if descriptor_dim < 1:
        raise ValueError("descriptor_dim must be positive")
    coefficients = StageALossCoefficients() if coefficients is None else coefficients
    return (
        coefficients.reconstruction * terms.reconstruction
        + coefficients.velocity * terms.velocity
        + coefficients.commitment * terms.commitment
        + (coefficients.plan_topology / descriptor_dim) * terms.plan_topology
        + coefficients.anchor * terms.anchor
    )


def combine_stage_b1(
    terms: StageB1LossTerms,
    descriptor_dim: int = 128,
    coefficients: StageB1LossCoefficients | None = None,
) -> Tensor:
    if descriptor_dim < 1:
        raise ValueError("descriptor_dim must be positive")
    coefficients = StageB1LossCoefficients() if coefficients is None else coefficients
    return (
        coefficients.diagram * terms.diagram
        + coefficients.autoregressive * terms.autoregressive
        + (coefficients.soft_plan / descriptor_dim) * terms.soft_plan
    )
