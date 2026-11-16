"""Checkpoint saving, loading, and compatibility validation."""

from __future__ import annotations

import hashlib
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim.optimizer import Optimizer

from .optimizer import PaperSchedule


class CheckpointBindingError(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckpointMetadata:
    stage: str
    architecture: str
    dataset: str
    skeleton: str
    statistics: str
    topology: str
    source: str
    parent: str | None


@dataclass(frozen=True)
class LoadedCheckpoint:
    update: int
    digest: str
    metadata: CheckpointMetadata
    resolved_config: dict[str, Any]
    manifests: dict[str, Any]
    selection: dict[str, Any] | None = None


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _rng_state(generators: dict[str, torch.Generator]) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "generators": {name: generator.get_state() for name, generator in generators.items()},
    }


def _restore_rng(state: dict[str, Any], generators: dict[str, torch.Generator]) -> None:
    expected = set(state["generators"])
    if set(generators) != expected:
        raise CheckpointBindingError(
            f"RNG stream mismatch: expected={sorted(expected)}, provided={sorted(generators)}"
        )
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        if not torch.cuda.is_available():
            raise CheckpointBindingError(
                "checkpoint contains CUDA RNG state but CUDA is unavailable"
            )
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    for name, generator in generators.items():
        generator.set_state(state["generators"][name])


def save_checkpoint_atomic(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    schedule: PaperSchedule,
    scaler: Any | None,
    update: int,
    metadata: CheckpointMetadata,
    generators: dict[str, torch.Generator],
    resolved_config: dict[str, Any],
    manifests: dict[str, Any],
    selection_state: dict[str, Any] | None = None,
) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "schedule": schedule.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "stage": metadata.stage,
        "update": update,
        "rng": _rng_state(generators),
        "resolved_config": resolved_config,
        "manifests": manifests,
        "metadata": asdict(metadata),
        "selection": selection_state,
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return _digest(destination)


def _validate_metadata(actual: CheckpointMetadata, expected: CheckpointMetadata) -> None:
    for field in asdict(expected):
        actual_value = getattr(actual, field)
        expected_value = getattr(expected, field)
        if actual_value != expected_value:
            if field == "parent":
                raise CheckpointBindingError(
                    f"parent checkpoint mismatch for {expected.stage}: "
                    f"expected {expected_value}, found {actual_value}"
                )
            raise CheckpointBindingError(
                f"{field} binding mismatch: expected {expected_value}, found {actual_value}"
            )


def load_checkpoint_bound(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    schedule: PaperSchedule,
    scaler: Any | None,
    generators: dict[str, torch.Generator],
    expected_metadata: CheckpointMetadata,
) -> LoadedCheckpoint:
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != 1:
        raise CheckpointBindingError("unsupported checkpoint format")
    actual_metadata = CheckpointMetadata(**payload["metadata"])
    _validate_metadata(actual_metadata, expected_metadata)
    if payload["stage"] != expected_metadata.stage:
        raise CheckpointBindingError("checkpoint stage does not match metadata")
    if (payload["scaler"] is None) != (scaler is None):
        raise CheckpointBindingError("gradient scaler configuration mismatch")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    schedule.load_state_dict(payload["schedule"])
    if scaler is not None:
        scaler.load_state_dict(payload["scaler"])
    _restore_rng(payload["rng"], generators)
    return LoadedCheckpoint(
        update=int(payload["update"]),
        digest=_digest(checkpoint_path),
        metadata=actual_metadata,
        resolved_config=payload["resolved_config"],
        manifests=payload["manifests"],
        selection=payload.get("selection"),
    )
