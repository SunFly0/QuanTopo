"""Fixed anatomical assignment and deterministic joint traversal."""

from __future__ import annotations

from collections.abc import Sequence

from quantopo.data.specs import PART_NAMES, DatasetSpec


def anatomical_part_indices(spec: DatasetSpec) -> tuple[int, ...]:
    assignment = [-1] * spec.joint_count
    for part_index, name in enumerate(PART_NAMES):
        for joint in spec.routing[name]:
            if assignment[joint] != -1:
                raise ValueError(f"joint {joint} has multiple anatomical assignments")
            assignment[joint] = part_index
    if any(index < 0 for index in assignment):
        raise ValueError("anatomical routing is not exhaustive")
    return tuple(assignment)


def depth_first_order(parents: Sequence[int]) -> tuple[int, ...]:
    roots = [joint for joint, parent in enumerate(parents) if parent == -1]
    if len(roots) != 1:
        raise ValueError("skeleton must contain exactly one root")
    children: list[list[int]] = [[] for _ in parents]
    for joint, parent in enumerate(parents):
        if parent >= 0:
            children[parent].append(joint)
    order: list[int] = []

    def visit(joint: int) -> None:
        order.append(joint)
        for child in sorted(children[joint]):
            visit(child)

    visit(roots[0])
    if len(order) != len(parents):
        raise ValueError("skeleton parent graph is disconnected or cyclic")
    return tuple(order)
