"""Dataset adapters, immutable skeleton specifications, and statistics."""

from .collate import collate_motion_records
from .dataset import MotionCaptionDataset, MotionRecord
from .normalization import RunningMoments
from .specs import DatasetSpec, get_dataset_spec

__all__ = [
    "DatasetSpec",
    "MotionCaptionDataset",
    "MotionRecord",
    "RunningMoments",
    "collate_motion_records",
    "get_dataset_spec",
]
