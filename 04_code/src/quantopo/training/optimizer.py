"""AdamW parameter groups and the training learning-rate schedule."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer

_NORMALIZATION_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.GroupNorm,
    nn.LayerNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
)


def build_adamw(
    module: nn.Module,
    *,
    learning_rate: float = 2e-4,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    weight_decay: float = 0.01,
) -> AdamW:
    if learning_rate <= 0 or epsilon <= 0 or weight_decay < 0:
        raise ValueError("invalid AdamW hyperparameters")
    no_decay_ids: set[int] = set()
    for submodule in module.modules():
        for parameter_name, parameter in submodule.named_parameters(recurse=False):
            if parameter.requires_grad and (
                parameter_name == "bias"
                or parameter.ndim <= 1
                or isinstance(submodule, _NORMALIZATION_TYPES)
            ):
                no_decay_ids.add(id(parameter))

    trainable = [
        (name, parameter)
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise ValueError("cannot build optimizer without trainable parameters")
    decay = [parameter for _, parameter in trainable if id(parameter) not in no_decay_ids]
    no_decay = [parameter for _, parameter in trainable if id(parameter) in no_decay_ids]
    owned = decay + no_decay
    if len(owned) != len({id(parameter) for parameter in owned}):
        raise RuntimeError("optimizer parameter ownership is not unique")
    if {id(parameter) for parameter in owned} != {id(parameter) for _, parameter in trainable}:
        raise RuntimeError("optimizer parameter ownership is not exhaustive")
    return AdamW(
        [
            {"params": decay, "weight_decay": weight_decay, "group_name": "decay"},
            {"params": no_decay, "weight_decay": 0.0, "group_name": "no_decay"},
        ],
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=epsilon,
    )


@dataclass
class PaperSchedule:
    total: int
    warmup: int = 2_000
    peak_learning_rate: float = 2e-4
    decay_fractions: tuple[float, ...] = (0.5, 0.75)
    decay_factor: float = 0.2
    last_update: int = -1

    def __post_init__(self) -> None:
        if self.total < 1 or self.warmup < 0 or self.warmup > self.total:
            raise ValueError("invalid schedule length or warmup")
        if self.peak_learning_rate <= 0 or not 0 < self.decay_factor <= 1:
            raise ValueError("invalid schedule learning rate or decay")
        if (
            not self.decay_fractions
            or tuple(sorted(self.decay_fractions)) != self.decay_fractions
            or any(not 0 < fraction < 1 for fraction in self.decay_fractions)
        ):
            raise ValueError("decay fractions must be sorted values between zero and one")

    def factor(self, update: int) -> float:
        if update < 0:
            raise ValueError("schedule update must be nonnegative")
        if self.warmup and update < self.warmup:
            return update / self.warmup
        factor = 1.0
        for fraction in self.decay_fractions:
            if update >= int(self.total * fraction):
                factor *= self.decay_factor
        return factor

    def apply(self, optimizer: Optimizer, update: int) -> float:
        learning_rate = self.peak_learning_rate * self.factor(update)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        self.last_update = update
        return learning_rate

    def state_dict(self) -> dict[str, int | float | tuple[float, ...]]:
        return {
            "total": self.total,
            "warmup": self.warmup,
            "peak_learning_rate": self.peak_learning_rate,
            "decay_fractions": self.decay_fractions,
            "decay_factor": self.decay_factor,
            "last_update": self.last_update,
        }

    def load_state_dict(self, state: dict[str, int | float | tuple[float, ...]]) -> None:
        for key in (
            "total",
            "warmup",
            "peak_learning_rate",
            "decay_fractions",
            "decay_factor",
        ):
            if state[key] != getattr(self, key):
                raise ValueError(f"schedule configuration mismatch: {key}")
        last_update = state["last_update"]
        if isinstance(last_update, tuple):
            raise ValueError("schedule last_update must be numeric")
        self.last_update = int(last_update)
