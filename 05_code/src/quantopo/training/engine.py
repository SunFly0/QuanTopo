"""Common single-update engine used by all QuanTopo training stages."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass as define_dataclass
from dataclasses import fields, is_dataclass, replace
from typing import TypeVar, cast

import torch
import torch.distributed as distributed
from torch import Tensor, nn

from quantopo.contracts.config import TrainingConfig

from .optimizer import PaperSchedule, build_adamw

T = TypeVar("T")


def move_to_device(value: T, device: torch.device | str) -> T:
    if isinstance(value, Tensor):
        return cast(T, value.to(device))
    if is_dataclass(value) and not isinstance(value, type):
        updates = {
            field.name: move_to_device(getattr(value, field.name), device)
            for field in fields(value)
        }
        return cast(T, replace(value, **updates))
    if isinstance(value, tuple):
        return cast(T, tuple(move_to_device(item, device) for item in value))
    if isinstance(value, list):
        return cast(T, [move_to_device(item, device) for item in value])
    if isinstance(value, Mapping):
        return cast(T, {key: move_to_device(item, device) for key, item in value.items()})
    return cast(T, value)


@define_dataclass(frozen=True)
class StepRecord:
    update: int
    loss: float
    learning_rate: float
    gradient_norm: float


@define_dataclass(frozen=True)
class EngineConfig:
    learning_rate: float = 2e-4
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 0.01
    warmup_updates: int = 2_000
    decay_fractions: tuple[float, ...] = (0.5, 0.75)
    decay_factor: float = 0.2
    gradient_clip: float = 1.0
    precision: str = "bf16"
    validate_every: int = 5_000

    @classmethod
    def from_training(cls, training: TrainingConfig) -> EngineConfig:
        optimizer = training.optimizer
        return cls(
            learning_rate=optimizer.learning_rate,
            beta1=optimizer.beta1,
            beta2=optimizer.beta2,
            epsilon=optimizer.epsilon,
            weight_decay=optimizer.weight_decay,
            warmup_updates=optimizer.warmup_steps,
            decay_fractions=optimizer.decay_fractions,
            decay_factor=optimizer.decay_factor,
            gradient_clip=optimizer.gradient_clip,
            precision=training.precision,
            validate_every=training.validate_every,
        )


class TrainingEngine:
    def __init__(
        self,
        model: nn.Module,
        *,
        total_updates: int,
        config: EngineConfig,
    ) -> None:
        if config.gradient_clip <= 0 or config.validate_every < 1:
            raise ValueError("gradient clip and validation cadence must be positive")
        if config.precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be fp32 or bf16")
        self.model = model
        self.optimizer = build_adamw(
            model,
            learning_rate=config.learning_rate,
            beta1=config.beta1,
            beta2=config.beta2,
            epsilon=config.epsilon,
            weight_decay=config.weight_decay,
        )
        self.schedule = PaperSchedule(
            total_updates,
            config.warmup_updates,
            config.learning_rate,
            config.decay_fractions,
            config.decay_factor,
        )
        self.gradient_clip = config.gradient_clip
        self.precision = config.precision
        self.validate_every = config.validate_every
        self.update = 0
        self.scaler = None

    def _reduced_loss(self, loss: Tensor) -> Tensor:
        detached = loss.detach().float()
        if distributed.is_available() and distributed.is_initialized():
            detached = detached.clone()
            distributed.all_reduce(detached)
            detached /= distributed.get_world_size()
        return detached

    def step(self, closure: Callable[[], Tensor]) -> StepRecord:
        self.schedule.apply(self.optimizer, self.update)
        self.optimizer.zero_grad(set_to_none=True)
        parameter = next(
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        )
        use_autocast = self.precision == "bf16" and parameter.device.type == "cuda"
        with torch.autocast(
            device_type=parameter.device.type,
            dtype=torch.bfloat16,
            enabled=use_autocast,
        ):
            loss = closure()
        if loss.ndim:
            raise ValueError("training closure must return a scalar loss")
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite training loss")
        torch.autograd.backward(loss)
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, self.gradient_clip)
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("non-finite gradient norm")
        self.optimizer.step()
        reduced = self._reduced_loss(loss)
        record = StepRecord(
            update=self.update,
            loss=float(reduced.item()),
            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
            gradient_norm=float(gradient_norm.item()),
        )
        self.update += 1
        return record

    def should_validate(self) -> bool:
        return self.update > 0 and self.update % self.validate_every == 0
