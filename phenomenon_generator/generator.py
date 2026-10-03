from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


class GenerationError(ValueError):
    pass


def validate_layout(values: list[int]) -> list[int]:
    nums = sorted(int(v) for v in values)
    if len(nums) != 7:
        raise GenerationError(f"7点である必要があります: {nums}")
    if len(set(nums)) != 7:
        raise GenerationError(f"重複があります: {nums}")
    bad = [n for n in nums if n < 1 or n > 37]
    if bad:
        raise GenerationError(f"1〜37の範囲外があります: {bad}")
    return nums


def gaps(values: list[int]) -> list[int]:
    nums = validate_layout(values)
    return [b - a for a, b in zip(nums, nums[1:])]


def _require_present(state: list[int], selected: list[int], label: str) -> None:
    missing = [n for n in selected if n not in state]
    if missing:
        raise GenerationError(f"{label}: 現在配置に存在しない点があります: {missing}")


def _translate(state: list[int], op: dict[str, Any]) -> list[int]:
    selected = [int(v) for v in op.get("values", state)]
    _require_present(state, selected, "移動")
    delta = int(op.get("delta", 0))
    offsets = {int(k): int(v) for k, v in op.get("offsets", {}).items()}

    out = []
    for n in state:
        if n in selected:
            out.append(n + delta + offsets.get(n, 0))
        else:
            out.append(n)
    return validate_layout(out)


def _deform(state: list[int], op: dict[str, Any]) -> list[int]:
    selected = [int(v) for v in op.get("values", state)]
    _require_present(state, selected, "変形")
    center = float(op["center"])
    scale = float(op.get("scale", 1.0))

    out = []
    for n in state:
        if n in selected:
            moved = int(round(center + (n - center) * scale))
            out.append(moved)
        else:
            out.append(n)
    return validate_layout(out)


def _reallocate(state: list[int], op: dict[str, Any]) -> list[int]:
    remove = [int(v) for v in op.get("remove", [])]
    add = [int(v) for v in op.get("add", [])]
    _require_present(state, remove, "再配置")

    out = [n for n in state if n not in remove]
    out.extend(add)
    return validate_layout(out)


def _rereference(
    state: list[int],
    op: dict[str, Any],
    history: list[list[int]],
) -> list[int]:
    source_index = int(op["source_index"])
    try:
        source = history[source_index]
    except IndexError as exc:
        raise GenerationError(f"再参照先がありません: source_index={source_index}") from exc

    source_values = [int(v) for v in op.get("source_values", source)]
    _require_present(source, source_values, "再参照元")

    remove_current = [int(v) for v in op.get("remove_current", [])]
    _require_present(state, remove_current, "再参照")
    delta = int(op.get("delta", 0))
    add_extra = [int(v) for v in op.get("add_extra", [])]

    out = [n for n in state if n not in remove_current]
    out.extend(n + delta for n in source_values)
    out.extend(add_extra)
    return validate_layout(out)


def apply_operation(
    state: list[int],
    operation: dict[str, Any],
    history: list[list[int]],
) -> list[int]:
    kind = operation.get("type")
    if kind == "translate":
        return _translate(state, operation)
    if kind == "deform":
        return _deform(state, operation)
    if kind == "reallocate":
        return _reallocate(state, operation)
    if kind == "rereference":
        return _rereference(state, operation, history)
    raise GenerationError(f"未知の操作です: {kind}")


def check_preserve(state: list[int], rule: dict[str, Any]) -> tuple[bool, str]:
    kind = rule.get("type")

    if kind == "points":
        values = [int(v) for v in rule.get("values", [])]
        missing = [n for n in values if n not in state]
        return (not missing, f"points missing={missing}")

    if kind == "region_count":
        lo, hi = int(rule["lo"]), int(rule["hi"])
        actual = sum(lo <= n <= hi for n in state)
        expected = int(rule["count"])
        return (actual == expected, f"region_count actual={actual} expected={expected}")

    if kind == "gaps":
        actual = gaps(state)
        expected = [int(v) for v in rule["values"]]
        return (actual == expected, f"gaps actual={actual} expected={expected}")

    if kind == "blank":
        lo, hi = int(rule["lo"]), int(rule["hi"])
        occupied = [n for n in state if lo <= n <= hi]
        return (not occupied, f"blank occupied={occupied}")

    if kind == "width":
        actual = max(state) - min(state)
        expected = int(rule["value"])
        tolerance = int(rule.get("tolerance", 0))
        ok = abs(actual - expected) <= tolerance
        return (
            ok,
            f"width actual={actual} expected={expected} tolerance={tolerance}",
        )

    raise GenerationError(f"未知の保存条件です: {kind}")


@dataclass
class StepResult:
    label: str
    before: list[int]
    after: list[int]
    operations: list[dict[str, Any]]
    preserve_checks: list[dict[str, Any]]


def run_recipe(recipe: dict[str, Any]) -> dict[str, Any]:
    initial = validate_layout(recipe["initial"])
    history: list[list[int]] = [initial]
    steps_out: list[dict[str, Any]] = []
    state = initial

    for i, step in enumerate(recipe.get("steps", []), start=1):
        before = state[:]
        op_log: list[dict[str, Any]] = []

        for op in step.get("operations", []):
            op_before = state[:]
            state = apply_operation(state, op, history)
            op_log.append(
                {
                    "operation": deepcopy(op),
                    "before": op_before,
                    "after": state[:],
                }
            )

        checks = []
        failed = []
        for rule in step.get("preserve", []):
            ok, detail = check_preserve(state, rule)
            item = {"rule": deepcopy(rule), "ok": ok, "detail": detail}
            checks.append(item)
            if not ok:
                failed.append(item)

        if failed:
            raise GenerationError(
                f"step {step.get('label', i)} の保存条件に違反しました: {failed}"
            )

        label = str(step.get("label", i))
        history.append(state[:])
        steps_out.append(
            StepResult(
                label=label,
                before=before,
                after=state[:],
                operations=op_log,
                preserve_checks=checks,
            ).__dict__
        )

    return {
        "name": recipe.get("name", "unnamed"),
        "purpose": recipe.get("purpose", ""),
        "initial": initial,
        "steps": steps_out,
        "final": state,
        "boundary": (
            "これは配置再現の操作記録であり、抽選原因・予測規則・因果機構を示さない。"
        ),
    }
