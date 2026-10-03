from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Mapping, Sequence


class PreservationError(ValueError):
    pass


@dataclass(frozen=True)
class PreservationRelation:
    consecutive: bool
    gaps: tuple[int, ...]


@dataclass(frozen=True)
class PreservationState:
    status: str  # active | released
    members: tuple[str, ...]
    relation: PreservationRelation
    remaining: int
    duration_unit: str = "successful_state_transition"

    def __post_init__(self) -> None:
        if self.status not in {"active", "released"}:
            raise PreservationError(f"invalid status: {self.status}")
        if self.status == "active" and self.remaining <= 0:
            raise PreservationError("active state requires remaining > 0")
        if self.status == "released" and self.remaining != 0:
            raise PreservationError("released state requires remaining == 0")
        if not self.relation.consecutive:
            raise PreservationError("v0.1 requires consecutive=true")
        if len(self.relation.gaps) != max(0, len(self.members) - 1):
            raise PreservationError("gap count must equal member_count - 1")


@dataclass(frozen=True)
class CandidateState:
    positions: Mapping[str, int]


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    reason: str
    next_preservation: PreservationState


def _ordered_positions(state: CandidateState, members: Sequence[str]) -> list[int]:
    missing = [m for m in members if m not in state.positions]
    if missing:
        raise PreservationError(f"missing mapped members: {missing}")
    return [int(state.positions[m]) for m in members]


def _all_points_sorted(state: CandidateState) -> list[tuple[str, int]]:
    return sorted(((k, int(v)) for k, v in state.positions.items()), key=lambda kv: kv[1])


def validate_transition(
    before: CandidateState,
    after: CandidateState,
    preservation: PreservationState,
    successor_map: Mapping[str, str],
) -> ValidationResult:
    if preservation.status == "released":
        return ValidationResult(True, "preservation already released", preservation)

    members = preservation.members

    missing_map = [m for m in members if m not in successor_map]
    if missing_map:
        return ValidationResult(False, f"missing successor mapping: {missing_map}", preservation)

    successors = [successor_map[m] for m in members]
    if len(set(successors)) != len(successors):
        return ValidationResult(False, "split/merge or duplicate successor mapping rejected", preservation)

    if any(s not in after.positions for s in successors):
        missing = [s for s in successors if s not in after.positions]
        return ValidationResult(False, f"mapped successor missing in after state: {missing}", preservation)

    before_positions = _ordered_positions(before, members)
    after_positions = _ordered_positions(after, successors)

    # v0.1 continuity/order: mapped members must preserve order
    if any(a >= b for a, b in zip(after_positions, after_positions[1:])):
        return ValidationResult(False, "mapped member order changed", preservation)

    expected_gaps = list(preservation.relation.gaps)
    actual_gaps = [b - a for a, b in zip(after_positions, after_positions[1:])]
    if actual_gaps != expected_gaps:
        return ValidationResult(
            False,
            f"gap mismatch: actual={actual_gaps} expected={expected_gaps}",
            preservation,
        )

    # v0.1 consecutive: no external point may sit between adjacent preserved successors.
    successor_set = set(successors)
    all_points = _all_points_sorted(after)
    pos_to_name = {pos: name for name, pos in all_points}
    for left, right in zip(after_positions, after_positions[1:]):
        between = [
            (name, pos)
            for name, pos in all_points
            if left < pos < right and name not in successor_set
        ]
        if between:
            return ValidationResult(
                False,
                f"external point inserted between preserved members: {between}",
                preservation,
            )

    remaining = preservation.remaining - 1
    if remaining == 0:
        next_state = replace(preservation, status="released", remaining=0)
    else:
        next_state = replace(preservation, remaining=remaining)

    return ValidationResult(True, "ACCEPT", next_state)


def explicit_pass(
    state: CandidateState,
    preservation: PreservationState,
    successor_map: Mapping[str, str] | None = None,
) -> ValidationResult:
    mapping = successor_map or {m: m for m in preservation.members}
    return validate_transition(state, state, preservation, mapping)
