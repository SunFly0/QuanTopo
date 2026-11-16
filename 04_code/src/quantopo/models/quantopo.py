"""Stage B2 training and end-to-end QuanTopo generation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from quantopo.contracts.tensors import (
    MotionBatch,
    NativeRealization,
    NormalizedRealization,
    PlanBatch,
    lengths_to_mask,
)
from quantopo.decoder.anatomical import DecodedMotion
from quantopo.flow.normalization import RealizationTransform
from quantopo.flow.objective import FlowMatchOutput, FlowObjective, RollInDecision, RollInGate
from quantopo.flow.sampler import euler_sample
from quantopo.flow.transformer import ConditionalFlow
from quantopo.models.planning import StageB1Model, freeze_module
from quantopo.representation.quantizer import QuantizedPlan
from quantopo.text.topology_predictor import PredictedPersistence


@dataclass(frozen=True)
class RollInBatch:
    plan: PlanBatch
    embeddings: Tensor
    native: NativeRealization
    decision: RollInDecision

    @property
    def proposed(self) -> Tensor:
        return self.decision.proposed

    @property
    def accepted(self) -> Tensor:
        return self.decision.accepted

    @property
    def clean_error(self) -> Tensor:
        return self.decision.clean_error

    @property
    def proposed_error(self) -> Tensor:
        return self.decision.proposed_error


@dataclass(frozen=True)
class StageB2Output:
    loss: Tensor
    objective: FlowMatchOutput | None
    rollin: RollInBatch
    regression: NormalizedRealization | None = None


@dataclass(frozen=True)
class GenerationOutput:
    plan: PlanBatch
    predicted_persistence: PredictedPersistence
    normalized: NormalizedRealization
    native: NativeRealization
    motion: DecodedMotion


def _per_sequence_mean(values: Tensor, mask: Tensor) -> Tensor:
    expanded = mask
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    weight = expanded.to(values.dtype).expand_as(values)
    numerator = (values * weight).flatten(start_dim=1).sum(dim=1)
    denominator = weight.flatten(start_dim=1).sum(dim=1).clamp_min(1.0)
    return numerator / denominator


def _reconstruction_error(
    planning: StageB1Model,
    batch: MotionBatch,
    plan_embeddings: Tensor,
    plan_lengths: Tensor,
    native: NativeRealization,
) -> Tensor:
    representation = planning.representation
    decoded = representation.decoder(plan_embeddings, plan_lengths, native)
    normalized_ric = (decoded.ric - representation.ric_mean) / representation.ric_std
    ric_error = F.smooth_l1_loss(normalized_ric, batch.ric, reduction="none")
    joint_error = (decoded.joints - batch.joints).abs() / representation.bone_reference
    return _per_sequence_mean(ric_error, batch.frame_mask) + _per_sequence_mean(
        joint_error, batch.frame_mask
    )


class StageB2Model(nn.Module):
    def __init__(
        self,
        *,
        planning: StageB1Model,
        transform: RealizationTransform,
        flow: ConditionalFlow | None,
        regressor: nn.Module | None = None,
        condition_dropout: float = 0.1,
        rollin_gate: RollInGate | None = None,
    ) -> None:
        super().__init__()
        if (flow is None) == (regressor is None):
            raise ValueError("provide exactly one realization generator")
        if flow is not None and flow.plan_dim != planning.representation.quantizer.embedding_dim:
            raise ValueError("flow plan dimension and representation codebook must agree")
        self.planning = planning
        self.transform = transform
        self.objective = (
            None if flow is None else FlowObjective(flow, condition_dropout=condition_dropout)
        )
        self.regressor = regressor
        self.rollin_gate = rollin_gate or RollInGate()
        freeze_module(self.planning)
        freeze_module(self.transform)

    @property
    def flow(self) -> ConditionalFlow:
        if self.objective is None:
            raise RuntimeError("deterministic realization variant has no flow")
        return self.objective.flow

    @property
    def realization_module(self) -> nn.Module:
        if self.objective is not None:
            return self.objective.flow
        if self.regressor is None:
            raise RuntimeError("realization generator is missing")
        return self.regressor

    def train(self, mode: bool = True) -> StageB2Model:
        super().train(mode)
        self.planning.eval()
        self.transform.eval()
        return self

    def _clean_pair(self, batch: MotionBatch) -> tuple[QuantizedPlan, NativeRealization]:
        with torch.no_grad():
            return self.planning.representation.encode(batch)

    def _rollin(
        self,
        batch: MotionBatch,
        update: int,
        *,
        proposal_generator: torch.Generator,
        ar_generator: torch.Generator,
    ) -> RollInBatch:
        clean_plan, clean_native = self._clean_pair(batch)
        clean_lengths = batch.plan_lengths
        with torch.no_grad():
            clean_error = _reconstruction_error(
                self.planning,
                batch,
                clean_plan.embeddings,
                clean_lengths,
                clean_native,
            )
            if update < self.rollin_gate.clean_steps:
                proposed_plan = PlanBatch(
                    clean_plan.indices,
                    clean_plan.mask,
                    clean_lengths,
                    self.planning.planner.codebook_size,
                )
                proposed_embeddings = clean_plan.embeddings
                proposed_native = clean_native
                proposed_error = clean_error
            else:
                text = self.planning.text_encoder(batch.captions)
                prediction = self.planning.topology_predictor(text, batch.lengths)
                proposed_plan = self.planning.planner.sample(
                    text,
                    prediction.descriptor,
                    self.planning.descriptor_scale,
                    batch.lengths,
                    generator=ar_generator,
                )
                codebook = self.planning.representation.quantizer.embedding
                proposed_embeddings = codebook[proposed_plan.indices]
                proposed_embeddings = proposed_embeddings * proposed_plan.mask.unsqueeze(-1)
                proposed_native = self.planning.representation.realization_encoder(
                    batch.ric,
                    batch.lengths,
                    proposed_embeddings,
                    proposed_plan.lengths,
                )
                proposed_error = _reconstruction_error(
                    self.planning,
                    batch,
                    proposed_embeddings,
                    proposed_plan.lengths,
                    proposed_native,
                )
            decision = self.rollin_gate.decide(
                update=update,
                clean_error=clean_error,
                proposed_error=proposed_error,
                generator=proposal_generator,
            )
            selector = decision.accepted[:, None]
            indices = torch.where(selector, proposed_plan.indices, clean_plan.indices)
            embeddings = torch.where(
                selector.unsqueeze(-1), proposed_embeddings, clean_plan.embeddings
            )
            native_values = torch.where(
                selector.unsqueeze(-1), proposed_native.values, clean_native.values
            )
            selected_plan = PlanBatch(
                indices=indices,
                mask=clean_plan.mask,
                lengths=clean_lengths,
                codebook_size=self.planning.planner.codebook_size,
            )
        return RollInBatch(
            plan=selected_plan,
            embeddings=embeddings,
            native=NativeRealization(native_values, batch.lengths),
            decision=decision,
        )

    def forward(
        self,
        batch: MotionBatch,
        *,
        update: int,
        noise_generator: torch.Generator,
        time_generator: torch.Generator,
        dropout_generator: torch.Generator,
        proposal_generator: torch.Generator,
        ar_generator: torch.Generator,
    ) -> StageB2Output:
        rollin = self._rollin(
            batch,
            update,
            proposal_generator=proposal_generator,
            ar_generator=ar_generator,
        )
        with torch.no_grad():
            text = self.planning.text_encoder(batch.captions)
            target = self.transform.normalize(rollin.native)
        if self.objective is not None:
            objective = self.objective(
                target,
                text,
                rollin.embeddings,
                rollin.plan.lengths,
                noise_generator=noise_generator,
                time_generator=time_generator,
                dropout_generator=dropout_generator,
            )
            return StageB2Output(loss=objective.loss, objective=objective, rollin=rollin)
        if self.regressor is None:
            raise RuntimeError("realization regressor is missing")
        prediction = self.regressor(
            text,
            rollin.embeddings,
            rollin.plan.lengths,
            batch.lengths,
        )
        loss = self.regressor.loss(prediction, target)  # type: ignore[attr-defined]
        return StageB2Output(
            loss=loss,
            objective=None,
            rollin=rollin,
            regression=prediction,
        )


class QuanTopo(nn.Module):
    def __init__(
        self,
        *,
        planning: StageB1Model,
        transform: RealizationTransform,
        flow: ConditionalFlow,
        euler_steps: int = 20,
    ) -> None:
        super().__init__()
        if euler_steps < 1:
            raise ValueError("euler_steps must be positive")
        if flow.plan_dim != planning.representation.quantizer.embedding_dim:
            raise ValueError("flow plan dimension and representation codebook must agree")
        self.planning = planning
        self.transform = transform
        self.flow = flow
        self.euler_steps = euler_steps
        freeze_module(self.planning)
        freeze_module(self.transform)

    @torch.no_grad()
    def generate(
        self,
        captions: tuple[str, ...],
        lengths: Tensor,
        *,
        ar_generator: torch.Generator,
        flow_generator: torch.Generator,
    ) -> GenerationOutput:
        if len(captions) != lengths.shape[0]:
            raise ValueError("captions and lengths batch sizes must agree")
        text = self.planning.text_encoder(captions)
        prediction = self.planning.topology_predictor(text, lengths)
        plan = self.planning.planner.sample(
            text,
            prediction.descriptor,
            self.planning.descriptor_scale,
            lengths,
            generator=ar_generator,
        )
        codebook = self.planning.representation.quantizer.embedding
        plan_embeddings = codebook[plan.indices] * plan.mask.unsqueeze(-1)
        frames = int(lengths.max().item())
        frame_mask = lengths_to_mask(lengths, frames)
        initial = torch.randn(
            lengths.shape[0],
            frames,
            128,
            dtype=plan_embeddings.dtype,
            device=lengths.device,
            generator=flow_generator,
        )
        initial = initial * frame_mask.unsqueeze(-1)
        normalized_values = euler_sample(
            self.flow,
            initial,
            steps=self.euler_steps,
            conditions={
                "text": text,
                "plan_embeddings": plan_embeddings,
                "plan_lengths": plan.lengths,
                "lengths": lengths,
            },
        )
        normalized = NormalizedRealization(normalized_values, lengths)
        native = self.transform.denormalize(normalized)
        motion = self.planning.representation.decoder(plan_embeddings, plan.lengths, native)
        return GenerationOutput(plan, prediction, normalized, native, motion)
