"""Stage A: topology-directed paired motion representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from quantopo.contracts.tensors import MotionBatch, NativeRealization
from quantopo.decoder.anatomical import AnatomicalDecoder, DecodedMotion
from quantopo.losses.reconstruction import motion_reconstruction_loss, physical_velocity_loss
from quantopo.losses.stages import StageALossTerms, combine_stage_a
from quantopo.models.factory import StageALossCoefficients
from quantopo.representation.encoders import PlanEncoder, RealizationEncoder
from quantopo.representation.plan_readout import PlanReadout
from quantopo.representation.quantizer import EMAQuantizer, QuantizedPlan


@dataclass(frozen=True)
class StageAOutput:
    q_star: QuantizedPlan
    native_r: NativeRealization
    anchor_input: NativeRealization
    full_reconstruction: DecodedMotion
    anchor_reconstruction: DecodedMotion
    plan_descriptor: Tensor
    terms: StageALossTerms
    loss: Tensor


class StageAModel(nn.Module):
    def __init__(
        self,
        plan_encoder: PlanEncoder,
        quantizer: EMAQuantizer,
        realization_encoder: RealizationEncoder,
        decoder: AnatomicalDecoder,
        plan_readout: PlanReadout,
        *,
        ric_mean: Tensor,
        ric_std: Tensor,
        loss_coefficients: StageALossCoefficients | None = None,
    ) -> None:
        super().__init__()
        if ric_mean.shape != (decoder.spec.ric_width,) or ric_std.shape != ric_mean.shape:
            raise ValueError("RIC statistics do not match decoder dataset")
        if bool(torch.any(ric_std <= 0)):
            raise ValueError("RIC standard deviations must be positive")
        self.plan_encoder = plan_encoder
        self.quantizer = quantizer
        self.realization_encoder = realization_encoder
        self.decoder = decoder
        self.plan_readout = plan_readout
        self.loss_coefficients = (
            StageALossCoefficients() if loss_coefficients is None else loss_coefficients
        )
        self.register_buffer("ric_mean", ric_mean.to(torch.float32).clone())
        self.register_buffer("ric_std", ric_std.to(torch.float32).clone())
        non_root_lengths = torch.linalg.vector_norm(decoder.offsets[1:], dim=-1)
        self.register_buffer("bone_reference", non_root_lengths.median().clamp_min(1e-6))

    def encode(self, batch: MotionBatch) -> tuple[QuantizedPlan, NativeRealization]:
        hidden, plan_mask = self.plan_encoder(batch.ric, batch.lengths)
        quantized = self.quantizer(hidden, plan_mask)
        native = self.realization_encoder(
            batch.ric,
            batch.lengths,
            quantized.embeddings,
            batch.plan_lengths,
        )
        return quantized, native

    def _normalized_ric(self, decoded: DecodedMotion) -> Tensor:
        return cast(Tensor, (decoded.ric - self.ric_mean) / self.ric_std)

    def forward(self, batch: MotionBatch, reference_descriptor: Tensor) -> StageAOutput:
        if batch.dataset != self.decoder.spec.name:
            raise ValueError("batch and decoder datasets differ")
        if reference_descriptor.shape != (batch.ric.shape[0], 128):
            raise ValueError("reference descriptor must have shape [B,128]")
        quantized, native = self.encode(batch)
        full = self.decoder(quantized.embeddings, batch.plan_lengths, native)
        anchor_input = NativeRealization(torch.zeros_like(native.values), native.lengths)
        anchor = self.decoder(quantized.embeddings, batch.plan_lengths, anchor_input)
        plan_descriptor = self.plan_readout(quantized.embeddings, quantized.mask)

        reconstruction = motion_reconstruction_loss(
            self._normalized_ric(full),
            batch.ric,
            full.joints,
            batch.joints,
            batch.frame_mask,
            self.bone_reference,
        )
        anchor_loss = motion_reconstruction_loss(
            self._normalized_ric(anchor),
            batch.ric,
            anchor.joints,
            batch.joints,
            batch.frame_mask,
            self.bone_reference,
        )
        velocity = physical_velocity_loss(
            full.joints, batch.joints, batch.frame_mask, self.bone_reference
        )
        topology = (plan_descriptor - reference_descriptor).square().sum(dim=-1).mean()
        terms = StageALossTerms(
            reconstruction,
            velocity,
            quantized.commitment_loss,
            topology,
            anchor_loss,
        )
        return StageAOutput(
            quantized,
            native,
            anchor_input,
            full,
            anchor,
            plan_descriptor,
            terms,
            combine_stage_a(terms, reference_descriptor.shape[-1], self.loss_coefficients),
        )
