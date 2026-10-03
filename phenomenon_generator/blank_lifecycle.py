from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


class BlankLifecycleError(ValueError):
    pass


@dataclass(frozen=True)
class BlankIntervalState:
    lo: int
    hi: int
    occupied: tuple[int, ...]

    @property
    def is_blank(self) -> bool:
        return len(self.occupied) == 0


@dataclass(frozen=True)
class BlankReentryObservation:
    blank_state: BlankIntervalState
    reentry_state: BlankIntervalState
    reentry_numbers: tuple[int, ...]
    persistent_numbers: tuple[int, ...]


def observe_blank_interval(layout: Sequence[int], lo: int, hi: int) -> BlankIntervalState:
    if lo > hi:
        raise BlankLifecycleError("lo must be <= hi")
    nums = sorted(int(v) for v in layout)
    if len(nums) != len(set(nums)):
        raise BlankLifecycleError("layout contains duplicate points")
    occupied = tuple(n for n in nums if lo <= n <= hi)
    return BlankIntervalState(int(lo), int(hi), occupied)


def observe_blank_then_reentry(
    blank_layout: Sequence[int],
    reentry_layout: Sequence[int],
    lo: int,
    hi: int,
    persistent_candidates: Iterable[int] = (),
) -> BlankReentryObservation:
    blank_state = observe_blank_interval(blank_layout, lo, hi)
    reentry_state = observe_blank_interval(reentry_layout, lo, hi)

    if not blank_state.is_blank:
        raise BlankLifecycleError("first state is not blank in the requested interval")

    reentry_numbers = reentry_state.occupied
    if not reentry_numbers:
        raise BlankLifecycleError("second state contains no reentry into the interval")

    a = set(int(x) for x in blank_layout)
    b = set(int(x) for x in reentry_layout)
    persistent = tuple(
        int(x) for x in persistent_candidates
        if int(x) in a and int(x) in b
    )

    return BlankReentryObservation(
        blank_state=blank_state,
        reentry_state=reentry_state,
        reentry_numbers=reentry_numbers,
        persistent_numbers=persistent,
    )
