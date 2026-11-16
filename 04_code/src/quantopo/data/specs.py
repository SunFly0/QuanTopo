"""Immutable HumanML3D and KIT-ML skeleton/routing definitions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

PART_NAMES = ("trunk", "head", "left_arm", "right_arm", "left_leg", "right_leg")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    display_name: str
    joint_count: int
    ric_width: int
    root_index: int
    parents: tuple[int, ...]
    frame_rate: float
    foot_indices: tuple[int, int, int, int]
    canonical_offsets_filename: str
    routing: Mapping[str, tuple[int, ...]]

    def __post_init__(self) -> None:
        if len(self.parents) != self.joint_count or self.parents[self.root_index] != -1:
            raise ValueError(f"invalid parent table for {self.display_name}")
        for joint, parent in enumerate(self.parents):
            if joint == self.root_index:
                continue
            if parent < 0 or parent >= self.joint_count or parent == joint:
                raise ValueError(f"invalid parent {parent} for joint {joint}")
        if tuple(self.routing) != PART_NAMES:
            raise ValueError("routing must use the fixed six-part order")
        flattened = [joint for name in PART_NAMES for joint in self.routing[name]]
        if sorted(flattened) != list(range(self.joint_count)):
            raise ValueError("routing must be disjoint and cover every joint")


def _routing(
    joint_count: int,
    *,
    head: tuple[int, ...],
    left_arm: tuple[int, ...],
    right_arm: tuple[int, ...],
    left_leg: tuple[int, ...],
    right_leg: tuple[int, ...],
) -> Mapping[str, tuple[int, ...]]:
    assigned = set(head + left_arm + right_arm + left_leg + right_leg)
    trunk = tuple(joint for joint in range(joint_count) if joint not in assigned)
    return MappingProxyType(
        {
            "trunk": trunk,
            "head": head,
            "left_arm": left_arm,
            "right_arm": right_arm,
            "left_leg": left_leg,
            "right_leg": right_leg,
        }
    )


_SPECS = {
    "humanml3d": DatasetSpec(
        name="humanml3d",
        display_name="HumanML3D",
        joint_count=22,
        ric_width=263,
        root_index=0,
        parents=(-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19),
        frame_rate=20.0,
        foot_indices=(7, 10, 8, 11),
        canonical_offsets_filename="canonical_offsets.npy",
        routing=_routing(
            22,
            head=(12, 15),
            left_arm=(13, 16, 18, 20),
            right_arm=(14, 17, 19, 21),
            left_leg=(1, 4, 7, 10),
            right_leg=(2, 5, 8, 11),
        ),
    ),
    "kitml": DatasetSpec(
        name="kitml",
        display_name="KIT-ML",
        joint_count=21,
        ric_width=251,
        root_index=0,
        parents=(-1, 0, 1, 2, 3, 3, 5, 6, 3, 8, 9, 0, 11, 12, 13, 14, 0, 16, 17, 18, 19),
        frame_rate=12.5,
        foot_indices=(19, 20, 14, 15),
        canonical_offsets_filename="canonical_offsets.npy",
        routing=_routing(
            21,
            head=(4,),
            left_arm=(8, 9, 10),
            right_arm=(5, 6, 7),
            left_leg=(16, 17, 18, 19, 20),
            right_leg=(11, 12, 13, 14, 15),
        ),
    ),
}


def get_dataset_spec(name: str) -> DatasetSpec:
    try:
        return _SPECS[name.lower()]
    except KeyError as error:
        raise ValueError(f"unsupported dataset: {name}") from error
