"""Persistence-target cache with source and configuration metadata."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

from quantopo.contracts.config import TopologyConfig
from quantopo.data.dataset import MotionViewIdentity

from .persistence import FixedPersistence, PersistenceDiagram

CACHE_VERSION = "quantopo-topology-v4-ric-view-bound"


class TopologyCacheMismatch(RuntimeError):
    """Raised when cached targets do not match their source or topology contract."""


@dataclass(frozen=True)
class CachedTopology:
    diagram: FixedPersistence
    descriptor: Tensor
    selected_indices: Tensor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".json")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


class TopologyCache:
    def __init__(
        self,
        root: Path,
        source_root: Path,
        split_path: Path,
        config: TopologyConfig,
        *,
        variant_fingerprint: str = "main",
    ) -> None:
        if not variant_fingerprint:
            raise ValueError("variant_fingerprint must be nonempty")
        self.root = Path(root)
        self.source_root = Path(source_root).resolve()
        self.split_path = Path(split_path).resolve()
        self.config = asdict(config)
        self.variant_fingerprint = variant_fingerprint
        self.manifest_path = self.root / "manifest.json"

    def _base_manifest(self) -> dict[str, Any]:
        return {
            "implementation": CACHE_VERSION,
            "split_sha256": _sha256(self.split_path),
            "topology": self.config,
            "topology_variant": self.variant_fingerprint,
            "entries": {},
        }

    def _manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return self._base_manifest()
        return cast(dict[str, Any], json.loads(self.manifest_path.read_text(encoding="utf-8")))

    def _validate_header(self, manifest: dict[str, Any]) -> None:
        if manifest.get("implementation") != CACHE_VERSION:
            raise TopologyCacheMismatch("cache implementation version changed")
        if manifest.get("topology") != self.config:
            raise TopologyCacheMismatch("topology configuration changed")
        if manifest.get("topology_variant") != self.variant_fingerprint:
            raise TopologyCacheMismatch("topology variant changed")
        if manifest.get("split_sha256") != _sha256(self.split_path):
            raise TopologyCacheMismatch("split source changed")

    def write(
        self,
        record_id: str,
        source_path: Path,
        selected_indices: Tensor,
        diagram: FixedPersistence,
        descriptor: Tensor,
        *,
        view_identity: MotionViewIdentity,
    ) -> None:
        source_path = Path(source_path).resolve()
        try:
            relative_source = source_path.relative_to(self.source_root).as_posix()
        except ValueError as error:
            raise ValueError("source path must be below source_root") from error
        manifest = self._manifest()
        self._validate_header(manifest)
        self.root.mkdir(parents=True, exist_ok=True)
        output_path = self.root / f"{record_id}.npz"
        descriptor_fd, temporary_name = tempfile.mkstemp(dir=self.root, suffix=".npz")
        os.close(descriptor_fd)
        try:
            np.savez(
                temporary_name,
                points=diagram.points.detach().cpu().numpy(),
                mask=diagram.mask.detach().cpu().numpy(),
                descriptor=descriptor.detach().cpu().numpy(),
                indices=selected_indices.detach().cpu().numpy(),
            )
            os.replace(temporary_name, output_path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        stat = source_path.stat()
        manifest["entries"][record_id] = {
            "source": relative_source,
            "source_bytes": stat.st_size,
            "source_sha256": _sha256(source_path),
            "output": output_path.name,
            "output_sha256": _sha256(output_path),
            "view": asdict(view_identity),
            "selected_indices": selected_indices.detach().cpu().tolist(),
        }
        _atomic_json(self.manifest_path, manifest)

    def load(
        self,
        record_id: str,
        source_path: Path,
        *,
        view_identity: MotionViewIdentity,
        expected_selected_indices: Tensor,
    ) -> CachedTopology:
        manifest = self._manifest()
        self._validate_header(manifest)
        try:
            entry = manifest["entries"][record_id]
        except KeyError as error:
            raise TopologyCacheMismatch(f"cache entry is missing: {record_id}") from error
        source_path = Path(source_path).resolve()
        expected_relative = source_path.relative_to(self.source_root).as_posix()
        stat = source_path.stat()
        if (
            entry["source"] != expected_relative
            or entry["source_bytes"] != stat.st_size
            or entry["source_sha256"] != _sha256(source_path)
        ):
            raise TopologyCacheMismatch("source motion changed")
        if entry.get("view") != asdict(view_identity):
            raise TopologyCacheMismatch("motion view changed")
        expected_indices = expected_selected_indices.detach().cpu().to(torch.long)
        if expected_indices.ndim != 1:
            raise ValueError("expected selected frame indices must be one-dimensional")
        if entry.get("selected_indices") != expected_indices.tolist():
            raise TopologyCacheMismatch("selected frame indices changed")
        output_path = self.root / entry["output"]
        if not output_path.exists() or entry["output_sha256"] != _sha256(output_path):
            raise TopologyCacheMismatch("cache output changed")
        with np.load(output_path, allow_pickle=False) as arrays:
            points = torch.from_numpy(arrays["points"].copy()).to(torch.float32)
            mask = torch.from_numpy(arrays["mask"].copy()).to(torch.bool)
            descriptor = torch.from_numpy(arrays["descriptor"].copy()).to(torch.float32)
            indices = torch.from_numpy(arrays["indices"].copy()).to(torch.long)
        if not torch.equal(indices.cpu(), expected_indices):
            raise TopologyCacheMismatch("cached selected frame indices changed")
        raw = PersistenceDiagram((), ())
        return CachedTopology(FixedPersistence(points, mask, raw), descriptor, indices)
