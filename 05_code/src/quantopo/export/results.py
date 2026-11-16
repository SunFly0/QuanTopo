"""Export generated motion and latent arrays with metadata."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from torch import Tensor

from quantopo.models.quantopo import GenerationOutput

_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ExportedResult:
    arrays_path: Path
    metadata_path: Path


def _numpy(value: Tensor) -> NDArray[Any]:
    return value.detach().cpu().numpy()


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_npz(path: Path, arrays: dict[str, NDArray[Any]]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _validate_metadata(
    output: GenerationOutput,
    captions: tuple[str, ...],
    lengths: Tensor,
    seeds: dict[str, int],
    checkpoint_digests: dict[str, str],
) -> None:
    batch = output.plan.indices.shape[0]
    if len(captions) != batch or lengths.shape != (batch,):
        raise ValueError("captions and lengths must match the generated batch")
    if not bool((lengths.cpu() == output.native.lengths.cpu()).all()):
        raise ValueError("export lengths differ from generated lengths")
    if not seeds or any(not name or not isinstance(seed, int) for name, seed in seeds.items()):
        raise ValueError("seeds must be a nonempty string-to-integer mapping")
    if not checkpoint_digests or any(
        not name or _SHA256.fullmatch(digest) is None for name, digest in checkpoint_digests.items()
    ):
        raise ValueError("checkpoint digests must be lowercase SHA-256 values")


def export_generation_result(
    output_dir: str | Path,
    *,
    output: GenerationOutput,
    captions: tuple[str, ...],
    lengths: Tensor,
    seeds: dict[str, int],
    checkpoint_digests: dict[str, str],
    stem: str = "generated_motion",
) -> ExportedResult:
    """Write all generated representations without serializing host-specific paths."""

    if not stem or Path(stem).name != stem:
        raise ValueError("result stem must be one path-free filename component")
    _validate_metadata(output, captions, lengths, seeds, checkpoint_digests)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    arrays_path = destination / f"{stem}.npz"
    metadata_path = destination / f"{stem}.json"
    arrays = {
        "ric": _numpy(output.motion.ric),
        "joints": _numpy(output.motion.joints),
        "plan_indices": _numpy(output.plan.indices),
        "plan_mask": _numpy(output.plan.mask),
        "persistence_points": _numpy(output.predicted_persistence.points),
        "persistence_existence": _numpy(output.predicted_persistence.existence),
        "descriptor": _numpy(output.predicted_persistence.descriptor),
        "native_realization": _numpy(output.native.values),
        "normalized_realization": _numpy(output.normalized.values),
        "lengths": _numpy(lengths),
    }
    if any(not np.isfinite(value).all() for value in arrays.values() if value.dtype.kind == "f"):
        raise ValueError("generated result contains non-finite values")
    _atomic_npz(arrays_path, arrays)
    payload: dict[str, object] = {
        "format_version": 1,
        "arrays": arrays_path.name,
        "arrays_sha256": _digest(arrays_path),
        "captions": list(captions),
        "lengths": lengths.detach().cpu().tolist(),
        "seeds": dict(sorted(seeds.items())),
        "checkpoint_digests": dict(sorted(checkpoint_digests.items())),
        "fields": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in arrays.items()
        },
    }
    _atomic_json(metadata_path, payload)
    return ExportedResult(arrays_path, metadata_path)
