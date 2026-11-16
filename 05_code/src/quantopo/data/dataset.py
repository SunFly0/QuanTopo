"""HumanML3D/KIT-ML caption-motion dataset with deterministic sampling."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from quantopo.contracts.config import DatasetConfig
from quantopo.data.specs import get_dataset_spec
from quantopo.geometry.ric import recover_joints_from_ric

from .assets import AssetManifest


@dataclass(frozen=True)
class MotionRecord:
    dataset: str
    motion_id: str
    caption: str
    ric: Tensor
    joints: Tensor

    def __post_init__(self) -> None:
        if self.ric.ndim != 2 or self.joints.ndim != 3 or self.joints.shape[-1] != 3:
            raise ValueError("record tensors must have shapes [T,D] and [T,J,3]")
        if self.ric.shape[0] != self.joints.shape[0]:
            raise ValueError("RIC and joints must have the same frame count")
        if self.ric.shape[0] < 4 or self.ric.shape[0] > 196:
            raise ValueError("a MotionRecord must contain 4..196 frames")
        if not bool(torch.isfinite(self.ric).all()) or not bool(torch.isfinite(self.joints).all()):
            raise ValueError("record tensors must be finite")


@dataclass(frozen=True)
class MotionViewIdentity:
    source_motion_id: str
    segment_start: int
    segment_end: int
    crop_start: int
    crop_end: int

    def __post_init__(self) -> None:
        if not self.source_motion_id:
            raise ValueError("source_motion_id must be nonempty")
        if not 0 <= self.segment_start < self.segment_end:
            raise ValueError("segment bounds must be nonempty and ordered")
        if not self.segment_start <= self.crop_start < self.crop_end <= self.segment_end:
            raise ValueError("crop bounds must be nonempty and lie within the segment")

    @property
    def frame_count(self) -> int:
        return self.crop_end - self.crop_start


@dataclass(frozen=True)
class _Sample:
    key: str
    source_motion_id: str
    captions: tuple[str, ...]
    start: int
    end: int | None


def _seed_from(*parts: object) -> int:
    encoded = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(encoded, digest_size=8).digest(), "little")


def _parse_captions(path: Path, frame_rate: float, motion_length: int) -> list[_Sample]:
    full: list[str] = []
    segmented: list[_Sample] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        fields = raw_line.split("#")
        if len(fields) < 4:
            raise ValueError(f"invalid caption line {path}:{line_number}")
        caption = fields[0].strip()
        try:
            start_seconds = float(fields[2])
            end_seconds = float(fields[3])
        except ValueError as error:
            raise ValueError(f"invalid caption times {path}:{line_number}") from error
        if start_seconds == 0.0 and end_seconds == 0.0:
            full.append(caption)
            continue
        start = max(0, int(round(start_seconds * frame_rate)))
        end = min(motion_length, int(round(end_seconds * frame_rate)))
        if end - start >= 4:
            segmented.append(
                _Sample(
                    key=f"{path.stem}@{start}:{end}:{line_number}",
                    source_motion_id=path.stem,
                    captions=(caption,),
                    start=start,
                    end=end,
                )
            )
    samples = segmented
    if full:
        samples.insert(
            0,
            _Sample(
                key=path.stem,
                source_motion_id=path.stem,
                captions=tuple(full),
                start=0,
                end=None,
            ),
        )
    if not samples:
        raise ValueError(f"no valid captions in {path}")
    return samples


class MotionCaptionDataset(Dataset[MotionRecord]):
    """One index per motion/valid segment, with one deterministic caption per epoch."""

    def __init__(
        self,
        config: DatasetConfig,
        manifest: AssetManifest,
        *,
        split: str,
        seed: int,
        max_frames: int = 196,
        validate_optional_joints: bool = False,
        joint_validation_tolerance: float = 1e-4,
    ) -> None:
        if manifest.split != split:
            raise ValueError(f"manifest split {manifest.split} does not match requested {split}")
        if max_frames != 196:
            raise ValueError("max_frames must be 196")
        if joint_validation_tolerance <= 0:
            raise ValueError("joint validation tolerance must be positive")
        self.config = config
        self.manifest = manifest
        self.seed = int(seed)
        self.epoch = 0
        self.max_frames = max_frames
        self.validate_optional_joints = validate_optional_joints
        self.joint_validation_tolerance = joint_validation_tolerance
        self.spec = get_dataset_spec(config.name)
        self.mean = torch.from_numpy(np.asarray(np.load(manifest.mean_path), dtype=np.float32))
        self.std = torch.from_numpy(np.asarray(np.load(manifest.std_path), dtype=np.float32))
        samples: list[_Sample] = []
        source_lengths: dict[str, int] = {}
        for motion_id in manifest.motion_ids:
            motion = np.load(manifest.motion_dir / f"{motion_id}.npy", mmap_mode="r")
            source_lengths[motion_id] = int(motion.shape[0])
            samples.extend(
                _parse_captions(
                    manifest.text_dir / f"{motion_id}.txt",
                    config.frame_rate,
                    int(motion.shape[0]),
                )
            )
        self.samples = tuple(samples)
        self.source_lengths = source_lengths

    def __len__(self) -> int:
        return len(self.samples)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = int(epoch)

    def source_ric_path(self, index: int) -> Path:
        """Return the RIC source backing a sampled motion view."""

        sample = self.samples[index]
        return self.manifest.motion_dir / f"{sample.source_motion_id}.npy"

    def source_joint_path(self, index: int) -> Path:
        """Return an optional validation-only joint source."""

        if self.manifest.joint_dir is None:
            raise FileNotFoundError("optional new_joints asset is unavailable")
        sample = self.samples[index]
        return self.manifest.joint_dir / f"{sample.source_motion_id}.npy"

    def view_identity(self, index: int) -> MotionViewIdentity:
        sample = self.samples[index]
        segment_start = sample.start
        segment_end = (
            self.source_lengths[sample.source_motion_id] if sample.end is None else sample.end
        )
        crop_start = segment_start
        if segment_end - segment_start > self.max_frames:
            crop_choices = segment_end - segment_start - self.max_frames + 1
            crop_start += _seed_from(sample.key, "crop") % crop_choices
        crop_end = min(segment_end, crop_start + self.max_frames)
        return MotionViewIdentity(
            source_motion_id=sample.source_motion_id,
            segment_start=segment_start,
            segment_end=segment_end,
            crop_start=crop_start,
            crop_end=crop_end,
        )

    def selected_global_indices(self, index: int, sample_count: int) -> Tensor:
        if sample_count < 1:
            raise ValueError("sample_count must be positive")
        view = self.view_identity(index)
        selected = min(view.frame_count, sample_count)
        if selected == view.frame_count:
            relative = torch.arange(view.frame_count)
        else:
            relative = torch.linspace(0, view.frame_count - 1, selected).round().long()
        return relative + view.crop_start

    def __getitem__(self, index: int) -> MotionRecord:
        sample = self.samples[index]
        caption_rng = random.Random(_seed_from(self.seed, self.epoch, index, sample.key, "caption"))
        caption = sample.captions[caption_rng.randrange(len(sample.captions))]
        motion_path = self.source_ric_path(index)
        view = self.view_identity(index)
        ric_array = np.asarray(
            np.load(motion_path)[view.crop_start : view.crop_end], dtype=np.float32
        )
        raw_ric = torch.from_numpy(ric_array.copy())
        ric = (raw_ric - self.mean) / self.std
        joints = recover_joints_from_ric(raw_ric.unsqueeze(0), self.spec)[0]
        if self.validate_optional_joints:
            joint_path = self.source_joint_path(index)
            optional = torch.from_numpy(
                np.asarray(
                    np.load(joint_path)[view.crop_start : view.crop_end],
                    dtype=np.float32,
                ).copy()
            )
            if optional.shape != joints.shape or not torch.allclose(
                optional,
                joints,
                rtol=0.0,
                atol=self.joint_validation_tolerance,
            ):
                raise ValueError(f"optional new_joints mismatch for {sample.source_motion_id}")
        return MotionRecord(
            dataset=self.config.name,
            motion_id=sample.key,
            caption=caption,
            ric=ric,
            joints=joints,
        )
