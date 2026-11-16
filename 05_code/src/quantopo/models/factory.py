"""Typed component choices shared by default and ablation composition roots."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageALossCoefficients:
    reconstruction: float = 1.0
    velocity: float = 0.5
    commitment: float = 0.25
    plan_topology: float = 0.1
    anchor: float = 0.5


@dataclass(frozen=True)
class StageB1LossCoefficients:
    diagram: float = 1.0
    autoregressive: float = 1.0
    soft_plan: float = 0.1


@dataclass(frozen=True)
class MethodComponents:
    stage_a_loss: StageALossCoefficients = StageALossCoefficients()
    stage_b1_loss: StageB1LossCoefficients = StageB1LossCoefficients()
    topology_modulation: bool = True
    topology_length_conditioning: bool = True
    planner_length_conditioning: bool = True
    decoder: str = "anatomical"
    quantizer: str = "ema_vq"
    plan_readout: str = "masked_mean"
    rollin_probability: float | None = None
    realization_generator: str = "conditional_flow"
    summary: str = "point_set"
    topology_sensitivity: str | None = None


def default_components() -> MethodComponents:
    return MethodComponents()
