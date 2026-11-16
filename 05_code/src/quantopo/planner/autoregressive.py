"""Causal text- and topology-conditioned discrete motion planner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from quantopo.contracts.tensors import PlanBatch, lengths_to_mask
from quantopo.planner.modulation import ZeroPreservingModulation
from quantopo.text.clip import TextFeatures


@dataclass(frozen=True)
class PlannerOutput:
    logits: Tensor
    states: Tensor
    mask: Tensor
    lengths: Tensor

    def __post_init__(self) -> None:
        if self.logits.ndim != 3:
            raise ValueError("planner logits must have shape [B,L,K]")
        if self.states.shape[:2] != self.logits.shape[:2]:
            raise ValueError("planner states must share the logits batch and length axes")
        if self.mask.shape != self.logits.shape[:2] or self.mask.dtype != torch.bool:
            raise ValueError("planner mask must be boolean with shape [B,L]")
        if self.lengths.shape != self.logits.shape[:1]:
            raise ValueError("planner lengths must have shape [B]")


class AutoregressivePlanner(nn.Module):
    def __init__(
        self,
        *,
        codebook_size: int = 512,
        text_width: int = 512,
        descriptor_dim: int = 128,
        layers: int = 6,
        width: int = 256,
        heads: int = 8,
        ffn: int = 1024,
        dropout: float = 0.1,
        max_positions: int = 64,
        modulation_hidden: int = 256,
        modulation_alpha: float = 0.1,
        modulation_clip: float = 5.0,
        modulation_enabled: bool = True,
        length_conditioned: bool = True,
    ) -> None:
        super().__init__()
        if min(codebook_size, text_width, descriptor_dim, layers, width, heads, ffn) < 1:
            raise ValueError("planner dimensions must be positive")
        if width % heads:
            raise ValueError("planner width must be divisible by attention heads")
        if max_positions < 1 or not 0 <= dropout < 1:
            raise ValueError("max_positions and dropout are invalid")
        self.codebook_size = codebook_size
        self.width = width
        self.max_positions = max_positions
        self.modulation_enabled = modulation_enabled
        self.length_conditioned = length_conditioned
        self.token_embedding = nn.Embedding(codebook_size, width)
        self.start = nn.Parameter(torch.randn(width) * 0.02)
        self.positions = nn.Parameter(torch.randn(max_positions, width) * 0.02)
        self.length_embedding = nn.Sequential(
            nn.Linear(1, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.text_projection = nn.Linear(text_width, width)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=ffn,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=layers, norm=nn.LayerNorm(width)
        )
        self.output = nn.Linear(width, codebook_size)
        self.modulation = ZeroPreservingModulation(
            state_width=width,
            descriptor_dim=descriptor_dim,
            hidden=modulation_hidden,
            vocabulary_size=codebook_size,
            alpha=modulation_alpha,
            clip_value=modulation_clip,
        )

    @staticmethod
    def plan_lengths(lengths: Tensor) -> Tensor:
        if lengths.ndim != 1 or lengths.dtype not in (torch.int32, torch.int64):
            raise ValueError("motion lengths must be a one-dimensional integer tensor")
        if bool(torch.any(lengths < 4)) or bool(torch.any(lengths > 196)):
            raise ValueError("motion lengths must lie in 4..196")
        return torch.div(lengths, 4, rounding_mode="floor")

    def _states(
        self,
        shifted_embeddings: Tensor,
        text: TextFeatures,
        lengths: Tensor,
        target_mask: Tensor,
    ) -> Tensor:
        sequence_length = shifted_embeddings.shape[1]
        if sequence_length > self.max_positions:
            raise ValueError(f"plan length exceeds {self.max_positions} learned positions")
        if text.tokens.shape[0] != shifted_embeddings.shape[0]:
            raise ValueError("text and plan batch sizes must agree")
        normalized_length = lengths.to(shifted_embeddings.dtype).unsqueeze(-1) / 196.0
        length_context = (
            self.length_embedding(normalized_length).unsqueeze(1)
            if self.length_conditioned
            else shifted_embeddings.new_zeros(shifted_embeddings.shape[0], 1, self.width)
        )
        target = shifted_embeddings + self.positions[:sequence_length].unsqueeze(0) + length_context
        causal_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=target.device),
            diagonal=1,
        )
        return cast(
            Tensor,
            self.decoder(
                target,
                self.text_projection(text.tokens),
                tgt_mask=causal_mask,
                tgt_key_padding_mask=~target_mask,
                memory_key_padding_mask=~text.token_mask,
            ),
        )

    def _logits(self, states: Tensor, descriptor: Tensor, descriptor_scale: Tensor) -> Tensor:
        base = self.output(states)
        if not self.modulation_enabled:
            return base
        return self.modulation.modulate(base, states, descriptor, descriptor_scale)

    def forward_teacher(
        self,
        text: TextFeatures,
        descriptor: Tensor,
        descriptor_scale: Tensor,
        lengths: Tensor,
        plan: PlanBatch,
    ) -> PlannerOutput:
        expected_lengths = self.plan_lengths(lengths)
        if not torch.equal(plan.lengths.to(expected_lengths.device), expected_lengths):
            raise ValueError("teacher-forced plan lengths must equal floor(T/4)")
        if plan.codebook_size != self.codebook_size:
            raise ValueError("teacher-forced plan uses a different codebook size")
        batch, sequence_length = plan.indices.shape
        shifted = self.start.view(1, 1, -1).expand(batch, sequence_length, -1).clone()
        if sequence_length > 1:
            shifted[:, 1:] = self.token_embedding(plan.indices[:, :-1])
        states = self._states(shifted, text, lengths, plan.mask)
        logits = self._logits(states, descriptor, descriptor_scale)
        logits = logits.masked_fill(~plan.mask.unsqueeze(-1), 0.0)
        states = states.masked_fill(~plan.mask.unsqueeze(-1), 0.0)
        return PlannerOutput(logits=logits, states=states, mask=plan.mask, lengths=plan.lengths)

    @torch.no_grad()
    def sample(
        self,
        text: TextFeatures,
        descriptor: Tensor,
        descriptor_scale: Tensor,
        lengths: Tensor,
        *,
        generator: torch.Generator,
    ) -> PlanBatch:
        plan_lengths = self.plan_lengths(lengths)
        maximum = int(plan_lengths.max().item())
        if maximum > self.max_positions:
            raise ValueError(f"plan length exceeds {self.max_positions} learned positions")
        batch = lengths.shape[0]
        indices = torch.zeros(batch, maximum, dtype=torch.long, device=lengths.device)
        for position in range(maximum):
            shifted = self.start.view(1, 1, -1).expand(batch, position + 1, -1).clone()
            if position:
                shifted[:, 1:] = self.token_embedding(indices[:, :position])
            active_mask = torch.ones(batch, position + 1, dtype=torch.bool, device=lengths.device)
            states = self._states(shifted, text, lengths, active_mask)
            logits = self._logits(states[:, -1:], descriptor, descriptor_scale)[:, 0]
            sampled = torch.multinomial(
                torch.softmax(logits, dim=-1), num_samples=1, generator=generator
            ).squeeze(-1)
            indices[:, position] = torch.where(position < plan_lengths, sampled, 0)
        mask = lengths_to_mask(plan_lengths, maximum)
        return PlanBatch(
            indices=indices,
            mask=mask,
            lengths=plan_lengths,
            codebook_size=self.codebook_size,
        )
