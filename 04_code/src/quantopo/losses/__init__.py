"""Losses and stage objective composition."""

from .diagram import DiagramLossTerms, SinkhornResult, augmented_sinkhorn_distance, diagram_loss
from .reconstruction import motion_reconstruction_loss, physical_velocity_loss
from .stages import (
    StageALossTerms,
    StageB1Loss,
    StageB1LossTerms,
    combine_stage_a,
    combine_stage_b1,
)

__all__ = [
    "DiagramLossTerms",
    "SinkhornResult",
    "StageALossTerms",
    "StageB1Loss",
    "StageB1LossTerms",
    "augmented_sinkhorn_distance",
    "combine_stage_a",
    "combine_stage_b1",
    "diagram_loss",
    "motion_reconstruction_loss",
    "physical_velocity_loss",
]
