"""Stage B1 composition for text-conditioned topology and plan learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from quantopo.contracts.tensors import MotionBatch, PersistenceBatch, PlanBatch
from quantopo.losses.diagram import DiagramLossTerms, diagram_loss
from quantopo.losses.stages import StageB1Loss, StageB1LossTerms, combine_stage_b1
from quantopo.models.factory import StageB1LossCoefficients
from quantopo.models.representation import StageAModel
from quantopo.planner.autoregressive import AutoregressivePlanner, PlannerOutput
from quantopo.representation.quantizer import QuantizedPlan, masked_sequence_mean
from quantopo.text.clip import FrozenCLIPTextEncoder, TextFeatures
from quantopo.text.topology_predictor import PredictedPersistence, TextTopologyPredictor


def freeze_module(module: nn.Module) -> None:
    """Freeze parameters and persistent state updates without disabling input gradients."""

    module.requires_grad_(False)
    module.eval()


@dataclass(frozen=True)
class StageB1Output:
    text: TextFeatures
    reference_plan: QuantizedPlan
    predicted_persistence: PredictedPersistence
    planner: PlannerOutput
    diagram_terms: DiagramLossTerms
    soft_descriptor: Tensor
    losses: StageB1Loss


def sequence_balanced_cross_entropy(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    """Compute token cross-entropy with equal weight for every sequence."""

    if logits.ndim != 3 or targets.shape != logits.shape[:2] or mask.shape != targets.shape:
        raise ValueError("logits, targets, and mask must have [B,L,V], [B,L], [B,L] shapes")
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    return masked_sequence_mean(token_loss, mask)


class StageB1Model(nn.Module):
    def __init__(
        self,
        *,
        representation: StageAModel,
        text_encoder: FrozenCLIPTextEncoder,
        topology_predictor: TextTopologyPredictor,
        planner: AutoregressivePlanner,
        descriptor_scale: Tensor,
        loss_coefficients: StageB1LossCoefficients | None = None,
    ) -> None:
        super().__init__()
        if descriptor_scale.shape != (128,):
            raise ValueError("descriptor_scale must have shape [128]")
        if bool(torch.any(descriptor_scale <= 0)):
            raise ValueError("descriptor_scale must be positive")
        if planner.codebook_size != representation.quantizer.codebook_size:
            raise ValueError("planner vocabulary and representation codebook must agree")
        if planner.token_embedding.embedding_dim != representation.quantizer.embedding_dim:
            raise ValueError("planner and representation plan dimensions must agree")
        self.representation = representation
        self.text_encoder = text_encoder
        self.topology_predictor = topology_predictor
        self.planner = planner
        self.loss_coefficients = (
            StageB1LossCoefficients() if loss_coefficients is None else loss_coefficients
        )
        self.register_buffer("descriptor_scale", descriptor_scale.float().clone().clamp_min(1e-3))
        freeze_module(self.representation)
        self.representation.quantizer.freeze()
        freeze_module(self.text_encoder)

    def train(self, mode: bool = True) -> StageB1Model:
        super().train(mode)
        self.representation.eval()
        self.representation.quantizer.freeze()
        self.text_encoder.eval()
        return self

    def _reference_plan(self, batch: MotionBatch) -> QuantizedPlan:
        with torch.no_grad():
            hidden, mask = self.representation.plan_encoder(batch.ric, batch.lengths)
            return cast(QuantizedPlan, self.representation.quantizer(hidden, mask))

    def _soft_plan_loss(
        self,
        planner_output: PlannerOutput,
        reference_descriptor: Tensor,
    ) -> tuple[Tensor, Tensor]:
        probabilities = torch.softmax(planner_output.logits, dim=-1)
        codebook = self.representation.quantizer.embedding.to(probabilities)
        expected_embeddings = probabilities @ codebook
        soft_descriptor = self.representation.plan_readout(expected_embeddings, planner_output.mask)
        loss = (soft_descriptor - reference_descriptor.detach()).square().sum(dim=-1).mean()
        return loss, soft_descriptor

    def forward(self, batch: MotionBatch, reference: PersistenceBatch) -> StageB1Output:
        if reference.points.shape[0] != batch.ric.shape[0]:
            raise ValueError("reference persistence batch size must match motion batch")
        reference_plan = self._reference_plan(batch)
        plan = PlanBatch(
            indices=reference_plan.indices,
            mask=reference_plan.mask,
            lengths=batch.plan_lengths,
            codebook_size=self.planner.codebook_size,
        )
        text = self.text_encoder(batch.captions)
        prediction = self.topology_predictor(text, batch.lengths)
        planner_output = self.planner.forward_teacher(
            text,
            prediction.descriptor,
            self.descriptor_scale,
            batch.lengths,
            plan,
        )
        diagram_terms = diagram_loss(
            prediction,
            reference,
            readout=self.topology_predictor.readout,
        )
        autoregressive = sequence_balanced_cross_entropy(
            planner_output.logits,
            reference_plan.indices,
            plan.mask,
        )
        soft_plan, soft_descriptor = self._soft_plan_loss(planner_output, reference.descriptor)
        terms = StageB1LossTerms(
            diagram=diagram_terms.total,
            autoregressive=autoregressive,
            soft_plan=soft_plan,
        )
        losses = StageB1Loss(
            diagram=terms.diagram,
            autoregressive=terms.autoregressive,
            soft_plan=terms.soft_plan,
            total=combine_stage_b1(
                terms,
                descriptor_dim=prediction.descriptor.shape[-1],
                coefficients=self.loss_coefficients,
            ),
        )
        return StageB1Output(
            text=text,
            reference_plan=reference_plan,
            predicted_persistence=prediction,
            planner=planner_output,
            diagram_terms=diagram_terms,
            soft_descriptor=soft_descriptor,
            losses=losses,
        )
