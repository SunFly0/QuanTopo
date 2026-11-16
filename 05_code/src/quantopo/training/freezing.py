"""Parameter ownership helpers shared by the three training stages."""

from __future__ import annotations

from collections.abc import Collection

from torch import nn


def freeze_module(module: nn.Module) -> None:
    module.requires_grad_(False)
    module.eval()


def unfreeze_module(module: nn.Module) -> None:
    module.requires_grad_(True)


def trainable_parameter_names(module: nn.Module) -> set[str]:
    return {name for name, parameter in module.named_parameters() if parameter.requires_grad}


def assert_parameter_ownership(module: nn.Module, expected_names: Collection[str]) -> set[str]:
    actual = trainable_parameter_names(module)
    expected = set(expected_names)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise RuntimeError(
            f"trainable parameter ownership mismatch: missing={missing}, unexpected={unexpected}"
        )
    if not actual:
        raise RuntimeError("training stage owns no parameters")
    return actual
