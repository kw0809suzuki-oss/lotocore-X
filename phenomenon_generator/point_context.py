from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


class PointContextError(ValueError):
    pass


@dataclass(frozen=True)
class PointContext:
    member: int
    present: bool
    left_neighbor: int | None
    right_neighbor: int | None
    left_gap: int | None
    right_gap: int | None


def observe_point_context(layout: Sequence[int], member: int) -> PointContext:
    nums = sorted(int(v) for v in layout)
    if len(nums) != len(set(nums)):
        raise PointContextError("layout contains duplicate points")

    m = int(member)
    if m not in nums:
        return PointContext(
            member=m,
            present=False,
            left_neighbor=None,
            right_neighbor=None,
            left_gap=None,
            right_gap=None,
        )

    i = nums.index(m)
    left = nums[i - 1] if i > 0 else None
    right = nums[i + 1] if i < len(nums) - 1 else None

    return PointContext(
        member=m,
        present=True,
        left_neighbor=left,
        right_neighbor=right,
        left_gap=(m - left) if left is not None else None,
        right_gap=(right - m) if right is not None else None,
    )


def context_changed(before: PointContext, after: PointContext) -> bool:
    if before.member != after.member:
        raise PointContextError("cannot compare different members")
    if not before.present or not after.present:
        return False

    return (
        before.left_neighbor,
        before.right_neighbor,
        before.left_gap,
        before.right_gap,
    ) != (
        after.left_neighbor,
        after.right_neighbor,
        after.left_gap,
        after.right_gap,
    )
