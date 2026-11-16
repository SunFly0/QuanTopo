"""Common realization-generator interfaces for ablation-only training."""

from __future__ import annotations

from typing import Protocol

from torch import Tensor

from quantopo.contracts.tensors import NormalizedRealization


class RealizationGenerator(Protocol):
    def predict(
        self,
        *args: object,
        **kwargs: object,
    ) -> NormalizedRealization: ...

    def objective(
        self, prediction: NormalizedRealization, target: NormalizedRealization
    ) -> Tensor: ...


def realization_kind(name: str) -> str:
    if name not in {"conditional_flow", "deterministic_regression"}:
        raise KeyError(f"unknown realization generator: {name}")
    return name
