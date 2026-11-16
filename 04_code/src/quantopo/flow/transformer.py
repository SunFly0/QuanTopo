"""Full-resolution conditional vector field with time AdaLN."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn

from quantopo.contracts.tensors import lengths_to_mask
from quantopo.representation.encoders import upsample_plan
from quantopo.text.clip import TextFeatures


def _sinusoidal_time(time: Tensor, width: int) -> Tensor:
    half = width // 2
    frequency = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=time.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    phase = time.float().unsqueeze(-1) * frequency.unsqueeze(0)
    embedding = torch.cat((phase.sin(), phase.cos()), dim=-1)
    if embedding.shape[-1] < width:
        embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
    return embedding


def _modulate(value: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return value * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class _FlowBlock(nn.Module):
    def __init__(self, width: int, heads: int, ffn: int, dropout: float) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(width)
        self.self_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn, width),
            nn.Dropout(dropout),
        )
        self.adaln = nn.Linear(width, 4 * width)

    def forward(
        self,
        state: Tensor,
        condition: Tensor,
        memory: Tensor,
        frame_padding: Tensor,
        memory_padding: Tensor,
    ) -> Tensor:
        self_shift, self_scale, ffn_shift, ffn_scale = self.adaln(condition).chunk(4, dim=-1)
        query = _modulate(self.self_norm(state), self_shift, self_scale)
        attended, _ = self.self_attention(
            query,
            query,
            query,
            key_padding_mask=frame_padding,
            need_weights=False,
        )
        state = state + attended
        attended, _ = self.cross_attention(
            self.cross_norm(state),
            memory,
            memory,
            key_padding_mask=memory_padding,
            need_weights=False,
        )
        state = state + attended
        state = state + self.ffn(_modulate(self.ffn_norm(state), ffn_shift, ffn_scale))
        return state.masked_fill(frame_padding.unsqueeze(-1), 0.0)


class ConditionalFlow(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 128,
        text_width: int = 512,
        plan_dim: int = 256,
        width: int = 384,
        blocks: int = 8,
        heads: int = 6,
        ffn: int = 1536,
        dropout: float = 0.1,
        max_frames: int = 196,
    ) -> None:
        super().__init__()
        if input_dim != 128:
            raise ValueError("flow coordinates must be 128-dimensional")
        if min(text_width, plan_dim, width, blocks, heads, ffn, max_frames) < 1:
            raise ValueError("flow dimensions must be positive")
        if width % heads or not 0 <= dropout < 1:
            raise ValueError("flow attention width or dropout is invalid")
        self.input_dim = input_dim
        self.plan_dim = plan_dim
        self.width = width
        self.max_frames = max_frames
        self.frame_projection = nn.Linear(input_dim, width)
        self.positions = nn.Parameter(torch.randn(max_frames, width) * 0.02)
        self.time_embedding = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.length_embedding = nn.Sequential(
            nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.text_projection = nn.Linear(text_width, width)
        self.plan_projection = nn.Linear(plan_dim, width)
        self.type_embedding = nn.Parameter(torch.randn(2, width) * 0.02)
        self.null_text = nn.Parameter(torch.randn(width) * 0.02)
        self.layers = nn.ModuleList(_FlowBlock(width, heads, ffn, dropout) for _ in range(blocks))
        self.output_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, input_dim)

    def _memory(
        self,
        text: TextFeatures,
        plan_embeddings: Tensor,
        plan_lengths: Tensor,
        lengths: Tensor,
        frames: int,
        text_drop_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        text_memory = self.text_projection(text.tokens) + self.type_embedding[0]
        text_mask = text.token_mask.clone()
        if bool(text_drop_mask.any()):
            text_memory = text_memory.clone()
            text_mask = text_mask.clone()
            text_memory[text_drop_mask] = 0.0
            text_memory[text_drop_mask, 0] = self.null_text
            text_mask[text_drop_mask] = False
            text_mask[text_drop_mask, 0] = True
        upsampled = upsample_plan(plan_embeddings, plan_lengths, lengths, frames=frames)
        plan_memory = self.plan_projection(upsampled) + self.type_embedding[1]
        frame_mask = lengths_to_mask(lengths, frames)
        memory = torch.cat((text_memory, plan_memory), dim=1)
        valid = torch.cat((text_mask, frame_mask), dim=1)
        return memory, ~valid

    def forward(
        self,
        state: Tensor,
        time: Tensor,
        text: TextFeatures,
        plan_embeddings: Tensor,
        plan_lengths: Tensor,
        lengths: Tensor,
        text_drop_mask: Tensor | None = None,
    ) -> Tensor:
        if state.ndim != 3 or state.shape[-1] != self.input_dim:
            raise ValueError(f"flow state must have shape [B,T,{self.input_dim}]")
        batch, frames, _ = state.shape
        if frames > self.max_frames:
            raise ValueError(f"flow input exceeds {self.max_frames} frames")
        if time.shape != (batch,):
            raise ValueError("flow time must have shape [B]")
        if plan_embeddings.ndim != 3 or plan_embeddings.shape[-1] != self.plan_dim:
            raise ValueError(f"plan embeddings must have shape [B,L,{self.plan_dim}]")
        if text_drop_mask is None:
            text_drop_mask = torch.zeros(batch, dtype=torch.bool, device=state.device)
        if text_drop_mask.shape != (batch,) or text_drop_mask.dtype != torch.bool:
            raise ValueError("text_drop_mask must be boolean with shape [B]")
        frame_mask = lengths_to_mask(lengths, frames)
        memory, memory_padding = self._memory(
            text,
            plan_embeddings,
            plan_lengths,
            lengths,
            frames,
            text_drop_mask,
        )
        condition = self.time_embedding(_sinusoidal_time(time, self.width))
        condition = condition + self.length_embedding(
            lengths.to(state.dtype).unsqueeze(-1) / self.max_frames
        )
        hidden = self.frame_projection(state) + self.positions[:frames]
        hidden = hidden.masked_fill(~frame_mask.unsqueeze(-1), 0.0)
        for layer in self.layers:
            hidden = layer(hidden, condition, memory, ~frame_mask, memory_padding)
        output = self.output(self.output_norm(hidden))
        return cast(Tensor, output * frame_mask.unsqueeze(-1).to(output.dtype))
