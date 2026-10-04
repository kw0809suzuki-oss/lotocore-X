from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from partial_gap_scope import find_partial_gap_matches


class StructureLifetimeError(ValueError):
    pass


@dataclass(frozen=True)
class StructurePresence:
    round: int
    present: bool
    matches: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class StructureLifetime:
    expected_gaps: tuple[int, ...]
    observations: tuple[StructurePresence, ...]
    consecutive_presence: int
    break_round: int | None


def observe_partial_gap_lifetime(
    rounds: Sequence[int],
    draws: Sequence[Sequence[int]],
    expected_gaps: Sequence[int],
) -> StructureLifetime:
    if len(rounds) != len(draws):
        raise StructureLifetimeError("round count must match draw count")
    if not rounds:
        raise StructureLifetimeError("at least one round is required")

    expected = tuple(int(x) for x in expected_gaps)
    observations: list[StructurePresence] = []

    for rnd, draw in zip(rounds, draws):
        matches = find_partial_gap_matches(draw, expected)
        observations.append(
            StructurePresence(
                round=int(rnd),
                present=bool(matches),
                matches=tuple(m.members for m in matches),
            )
        )

    run = 0
    break_round: int | None = None
    for obs in observations:
        if obs.present:
            run += 1
        else:
            break_round = obs.round
            break

    return StructureLifetime(
        expected_gaps=expected,
        observations=tuple(observations),
        consecutive_presence=run,
        break_round=break_round,
    )
