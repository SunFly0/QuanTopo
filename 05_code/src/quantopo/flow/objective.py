"""Flow-matching objective and reconstruction-gated roll-in decisions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from quantopo.contracts.tensors import NormalizedRealization, lengths_to_mask
from quantopo.text.clip import TextFeatures

from .transformer import ConditionalFlow


@dataclass(frozen=True)
class FlowMatchOutput:
    loss: Tensor
    noisy: Tensor
    target_velocity: Tensor
    predicted_velocity: Tensor
    noise: Tensor
    time: Tensor
    text_drop_mask: Tensor


class FlowObjective(nn.Module):
    def __init__(self, flow: ConditionalFlow, *, condition_dropout: float = 0.1) -> None:
        super().__init__()
        if not 0 <= condition_dropout <= 1:
            raise ValueError("condition_dropout must lie in [0,1]")
        self.flow = flow
        self.condition_dropout = condition_dropout

    def forward(
        self,
        target: NormalizedRealization,
        text: TextFeatures,
        plan_embeddings: Tensor,
        plan_lengths: Tensor,
        *,
        noise_generator: torch.Generator,
        time_generator: torch.Generator,
        dropout_generator: torch.Generator,
    ) -> FlowMatchOutput:
        values = target.values
        batch, frames, width = values.shape
        mask = lengths_to_mask(target.lengths, frames)
        noise = torch.randn(
            values.shape, dtype=values.dtype, device=values.device, generator=noise_generator
        )
        noise = noise * mask.unsqueeze(-1).to(noise.dtype)
        time = torch.rand(batch, dtype=values.dtype, device=values.device, generator=time_generator)
        text_drop_mask = (
            torch.rand(batch, device=values.device, generator=dropout_generator)
            < self.condition_dropout
        )
        interpolation = time[:, None, None]
        noisy = (1.0 - interpolation) * noise + interpolation * values
        target_velocity = values - noise
        predicted_velocity = self.flow(
            noisy,
            time,
            text,
            plan_embeddings,
            plan_lengths,
            target.lengths,
            text_drop_mask,
        )
        squared = (predicted_velocity - target_velocity).square()
        per_sequence = (squared * mask.unsqueeze(-1)).sum(dim=(1, 2))
        per_sequence = per_sequence / (target.lengths.to(squared.dtype) * width)
        return FlowMatchOutput(
            loss=per_sequence.mean(),
            noisy=noisy,
            target_velocity=target_velocity,
            predicted_velocity=predicted_velocity,
            noise=noise,
            time=time,
            text_drop_mask=text_drop_mask,
        )


@dataclass(frozen=True)
class RollInDecision:
    proposed: Tensor
    accepted: Tensor
    clean_error: Tensor
    proposed_error: Tensor


class RollInGate:
    def __init__(
        self,
        *,
        clean_steps: int = 50_000,
        proposal_probability: float = 0.2,
        acceptance_factor: float = 1.25,
    ) -> None:
        if clean_steps < 0 or not 0 <= proposal_probability <= 1 or acceptance_factor <= 0:
            raise ValueError("invalid roll-in policy")
        self.clean_steps = clean_steps
        self.proposal_probability = proposal_probability
        self.acceptance_factor = acceptance_factor

    def decide(
        self,
        *,
        update: int,
        clean_error: Tensor,
        proposed_error: Tensor,
        generator: torch.Generator,
    ) -> RollInDecision:
        if clean_error.ndim != 1 or proposed_error.shape != clean_error.shape:
            raise ValueError("roll-in errors must have matching shape [B]")
        if update < self.clean_steps:
            proposed = torch.zeros_like(clean_error, dtype=torch.bool)
        else:
            proposed = (
                torch.rand(
                    clean_error.shape,
                    device=clean_error.device,
                    generator=generator,
                )
                < self.proposal_probability
            )
        accepted = proposed & (proposed_error <= self.acceptance_factor * clean_error)
        return RollInDecision(proposed, accepted, clean_error, proposed_error)
