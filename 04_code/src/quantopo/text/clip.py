"""Frozen OpenCLIP text features loaded exclusively from a local checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TextFeatures:
    """Contextual token features and the pooled CLIP sentence feature."""

    tokens: Tensor
    token_mask: Tensor
    sentence: Tensor

    def __post_init__(self) -> None:
        if self.tokens.ndim != 3 or self.tokens.shape[-1] != 512:
            raise ValueError("text tokens must have shape [B,N,512]")
        if self.token_mask.shape != self.tokens.shape[:2] or self.token_mask.dtype != torch.bool:
            raise ValueError("token_mask must be boolean with shape [B,N]")
        if self.sentence.shape != (self.tokens.shape[0], 512):
            raise ValueError("sentence features must have shape [B,512]")
        if not bool(self.token_mask.any(dim=1).all()):
            raise ValueError("each caption must contain at least one valid token")


class TextBackbone(Protocol):
    width: int

    def __call__(self, captions: tuple[str, ...]) -> TextFeatures: ...


class _OpenCLIPBackbone(nn.Module):
    width = 512

    def __init__(self, model: nn.Module, tokenizer: object) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, captions: tuple[str, ...]) -> TextFeatures:
        token_ids = self.tokenizer(list(captions))  # type: ignore[operator]
        device = next(self.model.parameters()).device
        token_ids = token_ids.to(device)
        model = self.model

        x = model.token_embedding(token_ids)
        x = x + model.positional_embedding[: x.shape[1]].to(dtype=x.dtype)
        attention_mask = getattr(model, "attn_mask", None)
        x = model.transformer(x.permute(1, 0, 2), attn_mask=attention_mask).permute(1, 0, 2)
        tokens = model.ln_final(x)
        eot_positions = token_ids.argmax(dim=-1)
        sentence = tokens[torch.arange(tokens.shape[0], device=device), eot_positions]
        projection = getattr(model, "text_projection", None)
        if projection is not None:
            sentence = sentence @ projection
        if tokens.shape[-1] != self.width or sentence.shape[-1] != self.width:
            raise RuntimeError("OpenCLIP ViT-B/32 text features must be 512-dimensional")
        return TextFeatures(tokens=tokens, token_mask=token_ids.ne(0), sentence=sentence)


class FrozenCLIPTextEncoder(nn.Module):
    """A permanently frozen text encoder with an injectable test backbone."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        if getattr(backbone, "width", None) != 512:
            raise ValueError("CLIP text backbone width must be 512")
        self.backbone = backbone
        self.requires_grad_(False)
        super().train(False)

    @classmethod
    def from_open_clip_checkpoint(
        cls, checkpoint: str | Path, *, project_root: str | Path = "."
    ) -> FrozenCLIPTextEncoder:
        relative_checkpoint = Path(checkpoint)
        if relative_checkpoint.is_absolute() or ".." in relative_checkpoint.parts:
            raise ValueError("CLIP checkpoint path must be project-relative")
        root = Path(project_root).resolve()
        checkpoint_path = (root / relative_checkpoint).resolve()
        if checkpoint_path != root and root not in checkpoint_path.parents:
            raise ValueError("CLIP checkpoint path must remain within the project root")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"local CLIP checkpoint not found: {checkpoint_path}")
        try:
            import open_clip
        except ImportError as error:
            raise RuntimeError(
                "open_clip is required to load the local CLIP checkpoint; install open_clip_torch"
            ) from error

        model = open_clip.create_model("ViT-B-32", pretrained=None)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "state_dict" in payload:
            payload = payload["state_dict"]
        if not isinstance(payload, dict):
            raise RuntimeError("local CLIP checkpoint must contain a state dictionary")
        state = {
            key.removeprefix("module."): value
            for key, value in cast(dict[str, Tensor], payload).items()
        }
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "local CLIP checkpoint does not exactly match OpenCLIP ViT-B/32: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        return cls(_OpenCLIPBackbone(model, tokenizer))

    def train(self, mode: bool = True) -> FrozenCLIPTextEncoder:
        del mode
        super().train(False)
        self.backbone.train(False)
        return self

    def forward(self, captions: tuple[str, ...]) -> TextFeatures:
        if not captions:
            raise ValueError("captions must not be empty")
        self.backbone.train(False)
        with torch.no_grad():
            features = self.backbone(captions)
        return TextFeatures(
            tokens=features.tokens.detach(),
            token_mask=features.token_mask.detach(),
            sentence=features.sentence.detach(),
        )
