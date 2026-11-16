"""Validation of external processed-dataset assets without importing legacy code."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from quantopo.contracts.config import DatasetConfig


class AssetValidationError(ValueError):
    """Raised when configured dataset assets are incomplete or incompatible."""


@dataclass(frozen=True)
class AssetManifest:
    root: Path
    split: str
    motion_ids: tuple[str, ...]
    motion_dir: Path
    joint_dir: Path | None
    text_dir: Path
    mean_path: Path
    std_path: Path
    offsets_path: Path


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise AssetValidationError(f"missing {label}: {path}")


def validate_assets(
    config: DatasetConfig,
    *,
    split: str,
    project_root: Path | None = None,
) -> AssetManifest:
    base = Path.cwd() if project_root is None else Path(project_root)
    base = base.resolve()
    root = (base / config.root).resolve()
    try:
        root.relative_to(base)
    except ValueError as error:
        raise AssetValidationError("dataset root must stay below the project root") from error
    if not root.is_dir():
        raise AssetValidationError(f"dataset root is not a directory: {root}")
    split_path = root / f"{split}.txt"
    _require_file(split_path, f"split {split}")
    motion_dir = root / "new_joint_vecs"
    optional_joint_dir = root / "new_joints"
    joint_dir = optional_joint_dir if optional_joint_dir.is_dir() else None
    text_dir = root / "texts"
    for path, label in (
        (motion_dir, "new_joint_vecs directory"),
        (text_dir, "texts directory"),
    ):
        if not path.is_dir():
            raise AssetValidationError(f"missing {label}: {path}")

    mean_path = root / config.mean_path
    std_path = root / config.std_path
    offsets_path = root / config.offsets_path
    _require_file(mean_path, "Mean")
    _require_file(std_path, "Std")
    _require_file(offsets_path, "canonical offsets")
    mean = np.load(mean_path, mmap_mode="r")
    std = np.load(std_path, mmap_mode="r")
    offsets = np.load(offsets_path, mmap_mode="r")
    if mean.shape != (config.ric_width,):
        raise AssetValidationError(f"Mean must have width {config.ric_width}, got {mean.shape}")
    if std.shape != (config.ric_width,):
        raise AssetValidationError(f"Std must have width {config.ric_width}, got {std.shape}")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise AssetValidationError("Mean/Std must be finite and Std must be positive")
    if offsets.shape != (config.joint_count, 3) or not np.isfinite(offsets).all():
        raise AssetValidationError(
            f"canonical offsets must have shape ({config.joint_count}, 3) and be finite"
        )

    motion_ids = tuple(line.strip() for line in split_path.read_text().splitlines() if line.strip())
    if not motion_ids:
        raise AssetValidationError(f"split {split} is empty")
    if len(set(motion_ids)) != len(motion_ids):
        raise AssetValidationError(f"split {split} contains duplicate motion IDs")
    for motion_id in motion_ids:
        motion_path = motion_dir / f"{motion_id}.npy"
        text_path = text_dir / f"{motion_id}.txt"
        _require_file(motion_path, f"{motion_id} in new_joint_vecs")
        _require_file(text_path, f"{motion_id} in texts")
        motion = np.load(motion_path, mmap_mode="r")
        if motion.ndim != 2 or motion.shape[1] != config.ric_width:
            raise AssetValidationError(
                f"{motion_id} RIC must have shape [T,{config.ric_width}], got {motion.shape}"
            )
        if not text_path.read_text(encoding="utf-8").strip():
            raise AssetValidationError(f"{motion_id} caption file is empty")

    return AssetManifest(
        root=root,
        split=split,
        motion_ids=motion_ids,
        motion_dir=motion_dir,
        joint_dir=joint_dir,
        text_dir=text_dir,
        mean_path=mean_path,
        std_path=std_path,
        offsets_path=offsets_path,
    )
