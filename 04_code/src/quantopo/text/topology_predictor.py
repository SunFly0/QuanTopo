"""Text-and-length conditioned weighted persistence-diagram prediction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from quantopo.text.clip import TextFeatures
from quantopo.text.graph import TrunkStarMessagePassing
from quantopo.topology.readout import PersistenceReadout


@dataclass(frozen=True)
class PredictedPersistence:
    points: Tensor
    existence: Tensor
    descriptor: Tensor

    def __post_init__(self) -> None:
        if self.points.ndim != 4 or self.points.shape[1] != 2 or self.points.shape[-1] != 2:
            raise ValueError("predicted points must have shape [B,2,M,2]")
        if self.existence.shape != self.points.shape[:-1]:
            raise ValueError("existence must have shape [B,2,M]")
        if self.descriptor.shape != (self.points.shape[0], 128):
            raise ValueError("descriptor must have shape [B,128]")

    @property
    def birth(self) -> Tensor:
        return self.points[..., 0]

    @property
    def death(self) -> Tensor:
        return self.points[..., 1]


class _PointSetHead(nn.Module):
    def __init__(self, input_width: int, hidden: int, point_slots: int) -> None:
        super().__init__()
        self.point_slots = point_slots
        self.network = nn.Sequential(
            nn.Linear(input_width, hidden),
            nn.SiLU(),
            nn.Linear(hidden, point_slots * 3),
        )

    def forward(self, pooled: Tensor) -> tuple[Tensor, Tensor]:
        raw = self.network(pooled).reshape(pooled.shape[0], self.point_slots, 3)
        birth = F.softplus(raw[..., 0])
        death = birth + F.softplus(raw[..., 1]) + 1e-5
        points = torch.stack((birth, death), dim=-1)
        return points, torch.sigmoid(raw[..., 2])


class TextTopologyPredictor(nn.Module):
    def __init__(
        self,
        *,
        text_width: int = 512,
        query_count: int = 6,
        attention_heads: int = 8,
        graph_rounds: int = 2,
        point_slots: int = 16,
        head_hidden: int = 512,
        readout: PersistenceReadout,
        length_conditioned: bool = True,
    ) -> None:
        super().__init__()
        if text_width != 512 or query_count != 6:
            raise ValueError("the topology predictor requires width 512 and six part queries")
        if text_width % attention_heads:
            raise ValueError("text width must be divisible by attention heads")
        self.text_width = text_width
        self.length_conditioned = length_conditioned
        self.part_queries = nn.Parameter(torch.randn(query_count, text_width) * 0.02)
        self.length_embedding = nn.Sequential(
            nn.Linear(1, text_width),
            nn.SiLU(),
            nn.Linear(text_width, text_width),
        )
        self.cross_attention = nn.MultiheadAttention(text_width, attention_heads, batch_first=True)
        self.attention_norm = nn.LayerNorm(text_width)
        self.graph = TrunkStarMessagePassing(text_width, graph_rounds)
        pooled_width = 3 * text_width
        self.heads = nn.ModuleList(
            _PointSetHead(pooled_width, head_hidden, point_slots) for _ in range(2)
        )
        self.readout = readout
        self.readout.requires_grad_(False)
        self.readout.eval()

    def train(self, mode: bool = True) -> TextTopologyPredictor:
        super().train(mode)
        self.readout.eval()
        return self

    def forward(self, text: TextFeatures, lengths: Tensor) -> PredictedPersistence:
        batch = text.tokens.shape[0]
        if lengths.ndim != 1 or lengths.shape[0] != batch:
            raise ValueError(f"lengths must have shape [{batch}]")
        if lengths.dtype not in (torch.int32, torch.int64):
            raise TypeError("lengths must use an integer dtype")
        if bool(torch.any(lengths < 4)) or bool(torch.any(lengths > 196)):
            raise ValueError("supplied lengths must lie in 4..196")
        if text.tokens.shape[-1] != self.text_width:
            raise ValueError(f"text feature width must be {self.text_width}")

        normalized_length = lengths.to(text.tokens.dtype).unsqueeze(-1) / 196.0
        length_context = (
            self.length_embedding(normalized_length).unsqueeze(1)
            if self.length_conditioned
            else text.tokens.new_zeros(batch, 1, self.text_width)
        )
        queries = self.part_queries.unsqueeze(0).expand(batch, -1, -1) + length_context
        attended, _ = self.cross_attention(
            queries,
            text.tokens,
            text.tokens,
            key_padding_mask=~text.token_mask,
            need_weights=False,
        )
        queries = self.attention_norm(queries + attended)
        queries = self.graph(queries)
        pooled = torch.cat((queries.mean(dim=1), queries.amax(dim=1), text.sentence), dim=-1)

        point_sets: list[Tensor] = []
        masses: list[Tensor] = []
        for head in self.heads:
            points, existence = head(pooled)
            point_sets.append(points)
            masses.append(existence)
        all_points = torch.stack(point_sets, dim=1)
        all_masses = torch.stack(masses, dim=1)
        descriptor = self.readout(all_points, all_masses)
        return PredictedPersistence(all_points, all_masses, descriptor)
