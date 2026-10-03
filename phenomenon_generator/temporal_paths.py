from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


class TemporalPathError(ValueError):
    pass


@dataclass(frozen=True)
class TemporalPath:
    member: int
    states: tuple[bool, ...]

    def label(self) -> str:
        return "_".join("present" if x else "absent" for x in self.states)


@dataclass(frozen=True)
class TemporalPathObservation:
    rounds: tuple[int, ...]
    paths: tuple[TemporalPath, ...]

    def __post_init__(self) -> None:
        if len(self.rounds) < 2:
            raise TemporalPathError("temporal observation requires at least two rounds")
        if len(set(self.rounds)) != len(self.rounds):
            raise TemporalPathError("rounds must be unique")
        for path in self.paths:
            if len(path.states) != len(self.rounds):
                raise TemporalPathError("path length must match round window")

    def path_types(self) -> tuple[str, ...]:
        return tuple(path.label() for path in self.paths)

    def has_coexisting_path_types(self) -> bool:
        return len(set(self.path_types())) > 1


def observe_paths(
    rounds: Sequence[int],
    draws: Sequence[Iterable[int]],
    members: Sequence[int],
) -> TemporalPathObservation:
    if len(rounds) != len(draws):
        raise TemporalPathError("round count must match draw count")
    draw_sets = [set(int(x) for x in draw) for draw in draws]
    paths = tuple(
        TemporalPath(
            member=int(member),
            states=tuple(int(member) in draw for draw in draw_sets),
        )
        for member in members
    )
    return TemporalPathObservation(tuple(int(r) for r in rounds), paths)


def match_required_paths(
    observation: TemporalPathObservation,
    required: Mapping[int, Sequence[bool]],
) -> bool:
    actual = {p.member: p.states for p in observation.paths}
    for member, states in required.items():
        if int(member) not in actual:
            return False
        if actual[int(member)] != tuple(bool(x) for x in states):
            return False
    return True
