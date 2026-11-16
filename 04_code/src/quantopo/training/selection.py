"""Validation cadence and checkpoint-selection primitives."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from typing import Literal, TypeVar


@dataclass(frozen=True)
class SelectionMetric:
    name: str
    mode: Literal["min", "max"]

    def __post_init__(self) -> None:
        if not self.name or self.mode not in {"min", "max"}:
            raise ValueError("selection metric needs a name and min/max mode")


@dataclass
class CheckpointSelector:
    metric: SelectionMetric
    best_value: float | None = None
    best_update: int | None = None

    def observe(self, update: int, value: float) -> bool:
        if update < 1:
            raise ValueError("selection updates are one-based")
        better = self.best_value is None or (
            value < self.best_value if self.metric.mode == "min" else value > self.best_value
        )
        if better:
            self.best_value = float(value)
            self.best_update = update
        return better

    def state_dict(self) -> dict[str, object]:
        return {
            "metric": asdict(self.metric),
            "best_value": self.best_value,
            "best_update": self.best_update,
        }


T = TypeVar("T")


def run_training_updates(
    batches: Iterator[T],
    *,
    total_updates: int,
    current_update: Callable[[], int],
    train_step: Callable[[T], object],
    validate_every: int,
    checkpoint_callback: Callable[[int, str], None],
    validation_callback: Callable[[int], float] | None = None,
    selector: CheckpointSelector | None = None,
) -> None:
    """Run updates and expose periodic/latest/best boundaries without stage coupling."""

    if total_updates < 1 or validate_every < 1:
        raise ValueError("update budget and validation cadence must be positive")
    if (validation_callback is None) != (selector is None):
        raise ValueError("validation callback and selector must be provided together")
    for batch in batches:
        train_step(batch)
        update = current_update()
        if update % validate_every == 0 or update >= total_updates:
            checkpoint_callback(update, "latest")
            checkpoint_callback(update, f"update_{update:07d}")
            if validation_callback is not None and selector is not None:
                value = validation_callback(update)
                if selector.observe(update, value):
                    checkpoint_callback(update, "best")
        if update >= total_updates:
            return
