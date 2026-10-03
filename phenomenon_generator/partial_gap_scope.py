from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


class PartialGapError(ValueError):
    pass


@dataclass(frozen=True)
class PartialGapMatch:
    members: tuple[int, ...]
    gaps: tuple[int, ...]
    start_index: int
    end_index: int


def _sorted_unique(values: Iterable[int]) -> list[int]:
    nums = sorted(int(v) for v in values)
    if len(nums) != len(set(nums)):
        raise PartialGapError("layout contains duplicate points")
    return nums


def gaps_of_members(members: Sequence[int]) -> tuple[int, ...]:
    nums = _sorted_unique(members)
    if len(nums) < 2:
        raise PartialGapError("at least two members are required")
    return tuple(b - a for a, b in zip(nums, nums[1:]))


def find_partial_gap_matches(
    layout: Sequence[int],
    expected_gaps: Sequence[int],
) -> tuple[PartialGapMatch, ...]:
    nums = _sorted_unique(layout)
    expected = tuple(int(x) for x in expected_gaps)
    if not expected:
        raise PartialGapError("expected_gaps must not be empty")

    width = len(expected) + 1
    matches: list[PartialGapMatch] = []
    for i in range(0, len(nums) - width + 1):
        members = tuple(nums[i : i + width])
        actual = gaps_of_members(members)
        if actual == expected:
            matches.append(
                PartialGapMatch(
                    members=members,
                    gaps=actual,
                    start_index=i,
                    end_index=i + width - 1,
                )
            )
    return tuple(matches)


def translated_subset(
    before_members: Sequence[int],
    after_members: Sequence[int],
) -> tuple[bool, int | None]:
    a = _sorted_unique(before_members)
    b = _sorted_unique(after_members)
    if len(a) != len(b) or not a:
        return False, None

    deltas = [y - x for x, y in zip(a, b)]
    if len(set(deltas)) != 1:
        return False, None
    return True, deltas[0]
