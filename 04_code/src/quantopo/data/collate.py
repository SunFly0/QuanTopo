"""Right-padding collator for motion records."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from quantopo.contracts.tensors import MotionBatch

from .dataset import MotionRecord
from .specs import get_dataset_spec


def collate_motion_records(records: Sequence[MotionRecord]) -> MotionBatch:
    if not records:
        raise ValueError("cannot collate an empty record sequence")
    dataset = records[0].dataset
    if any(record.dataset != dataset for record in records):
        raise ValueError("all records in a batch must use the same dataset")
    spec = get_dataset_spec(dataset)
    batch = len(records)
    max_frames = max(record.ric.shape[0] for record in records)
    ric = torch.zeros(batch, max_frames, spec.ric_width, dtype=records[0].ric.dtype)
    joints = torch.zeros(
        batch,
        max_frames,
        spec.joint_count,
        3,
        dtype=records[0].joints.dtype,
    )
    lengths = torch.empty(batch, dtype=torch.long)
    for index, record in enumerate(records):
        frames = record.ric.shape[0]
        if record.ric.shape[1] != spec.ric_width:
            raise ValueError(f"record {record.motion_id} has the wrong RIC width")
        if record.joints.shape[1:] != (spec.joint_count, 3):
            raise ValueError(f"record {record.motion_id} has the wrong joint shape")
        ric[index, :frames] = record.ric
        joints[index, :frames] = record.joints
        lengths[index] = frames
    return MotionBatch.from_tensors(
        dataset=dataset,
        ric=ric,
        joints=joints,
        lengths=lengths,
        captions=[record.caption for record in records],
        motion_ids=[record.motion_id for record in records],
    )
