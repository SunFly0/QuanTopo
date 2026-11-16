"""Versioned, parent-bound realization-statistics artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from quantopo.flow.normalization import RealizationTransform

FORMAT_VERSION = "quantopo-realization-statistics-v1"


class StatisticsBindingError(RuntimeError):
    """Raised when statistics do not belong to the selected training lineage."""


@dataclass(frozen=True)
class StatisticsBindings:
    parent: str
    split: str
    split_digest: str
    architecture: str
    dataset: str
    ric_statistics: str
    source: str


@dataclass(frozen=True)
class StatisticsMetadata:
    format_version: str
    parent: str
    split: str
    split_digest: str
    architecture: str
    dataset: str
    ric_statistics: str
    source: str
    mean_shape: tuple[int, ...]
    scale_shape: tuple[int, ...]
    mean_dtype: str
    scale_dtype: str
    npz_sha256: str

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> StatisticsMetadata:
        return cls(
            **{
                **value,
                "mean_shape": tuple(value["mean_shape"]),
                "scale_shape": tuple(value["scale_shape"]),
            }
        )


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _metadata_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def _array(value: Tensor | NDArray[Any]) -> NDArray[np.float32]:
    if isinstance(value, Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def save_realization_statistics(
    path: str | Path,
    mean: Tensor | NDArray[Any],
    scale: Tensor | NDArray[Any],
    bindings: StatisticsBindings,
) -> StatisticsMetadata:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mean_array, scale_array = _array(mean), _array(scale)
    descriptor, temporary_name = tempfile.mkstemp(dir=destination.parent, suffix=".npz")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, mean=mean_array, scale=scale_array)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    metadata = StatisticsMetadata(
        format_version=FORMAT_VERSION,
        **asdict(bindings),
        mean_shape=mean_array.shape,
        scale_shape=scale_array.shape,
        mean_dtype=str(mean_array.dtype),
        scale_dtype=str(scale_array.dtype),
        npz_sha256=_digest(destination),
    )
    metadata_path = _metadata_path(destination)
    descriptor, temporary_name = tempfile.mkstemp(dir=destination.parent, suffix=".json")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(asdict(metadata), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, metadata_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return metadata


def load_realization_statistics(
    path: str | Path,
    expected_bindings: StatisticsBindings,
) -> RealizationTransform:
    source = Path(path)
    metadata_path = _metadata_path(source)
    if not source.is_file() or not metadata_path.is_file():
        raise StatisticsBindingError("statistics NPZ or metadata is missing")
    metadata = StatisticsMetadata.from_json(json.loads(metadata_path.read_text(encoding="utf-8")))
    if metadata.format_version != FORMAT_VERSION:
        raise StatisticsBindingError("unsupported statistics format version")
    for name, expected in asdict(expected_bindings).items():
        actual = getattr(metadata, name)
        if actual != expected:
            raise StatisticsBindingError(
                f"statistics {name} binding mismatch: expected {expected}, found {actual}"
            )
    if metadata.npz_sha256 != _digest(source):
        raise StatisticsBindingError("statistics NPZ digest mismatch")
    with np.load(source, allow_pickle=False) as arrays:
        mean_array = np.asarray(arrays["mean"])
        scale_array = np.asarray(arrays["scale"])
    if (
        mean_array.shape != metadata.mean_shape
        or scale_array.shape != metadata.scale_shape
        or str(mean_array.dtype) != metadata.mean_dtype
        or str(scale_array.dtype) != metadata.scale_dtype
    ):
        raise StatisticsBindingError("statistics shape or dtype metadata mismatch")
    return RealizationTransform(
        torch.from_numpy(mean_array.astype(np.float32, copy=True)),
        torch.from_numpy(scale_array.astype(np.float32, copy=True)),
    )
