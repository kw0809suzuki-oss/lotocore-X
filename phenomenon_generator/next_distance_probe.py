from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd

from partial_gap_scope import find_partial_gap_matches, translated_subset
from point_context import context_changed, observe_point_context


NUM_COLS = [f"n{i}" for i in range(1, 8)]


@dataclass(frozen=True)
class Detection:
    structure_id: str
    carriers: tuple[int, ...]
    detail: str


def _draw(row: pd.Series) -> list[int]:
    return sorted(int(row[c]) for c in NUM_COLS)


def _union_members(matches: Iterable[Iterable[int]]) -> tuple[int, ...]:
    return tuple(sorted({int(x) for group in matches for x in group}))


def detect_s1(prev: Sequence[int], cur: Sequence[int]) -> Detection | None:
    # Exact signature v0: the [1,1] three-point local structure is present
    # in at least two consecutive observed draws. Absolute position is free.
    pm = find_partial_gap_matches(prev, [1, 1])
    cm = find_partial_gap_matches(cur, [1, 1])
    if not pm or not cm:
        return None
    carriers = _union_members(m.members for m in cm)
    return Detection("S1", carriers, "gap[1,1] present in consecutive draws")


def detect_s2(a: Sequence[int], b: Sequence[int], c: Sequence[int]) -> Detection | None:
    # Exact signature v0: at least one present-absent-present member and
    # at least one present-present-present member coexist in the same 3-draw window.
    sa, sb, sc = set(a), set(b), set(c)
    pap = sorted((sa & sc) - sb)
    ppp = sorted(sa & sb & sc)
    if not pap or not ppp:
        return None
    carriers = tuple(sorted(set(pap) | set(ppp)))
    return Detection("S2", carriers, f"PAP={pap}; PPP={ppp}")


def detect_s3(a: Sequence[int], b: Sequence[int], c: Sequence[int]) -> Detection | None:
    # Exact signature v0: a point persists through 3 draws while its local
    # left/right context changes at least once.
    common = sorted(set(a) & set(b) & set(c))
    carriers: list[int] = []
    for n in common:
        ca = observe_point_context(a, n)
        cb = observe_point_context(b, n)
        cc = observe_point_context(c, n)
        if context_changed(ca, cb) or context_changed(cb, cc):
            carriers.append(n)
    if not carriers:
        return None
    return Detection("S3", tuple(carriers), "3-draw persistent point with mutable local context")


def _largest_internal_gap(draw: Sequence[int]) -> tuple[int, int] | None:
    xs = sorted(int(x) for x in draw)
    if len(xs) < 2:
        return None
    gap, left, right = max((b - a, a, b) for a, b in zip(xs, xs[1:]))
    if gap <= 1:
        return None
    return left + 1, right - 1


def detect_s4(a: Sequence[int], b: Sequence[int], c: Sequence[int]) -> Detection | None:
    # Exact signature v0: the largest internal blank interval in the middle draw
    # had occupancy in the prior draw, is re-entered in the current draw,
    # and at least one point persists from middle->current outside that interval.
    interval = _largest_internal_gap(b)
    if interval is None:
        return None
    lo, hi = interval
    prior_inside = sorted(n for n in a if lo <= n <= hi)
    current_inside = sorted(n for n in c if lo <= n <= hi)
    persistent_outside = sorted(
        n for n in (set(b) & set(c)) if not (lo <= n <= hi)
    )
    if not prior_inside or not current_inside or not persistent_outside:
        return None
    carriers = tuple(sorted(set(current_inside) | set(persistent_outside)))
    return Detection(
        "S4",
        carriers,
        f"blank=[{lo},{hi}]; reentry={current_inside}; persistent_outside={persistent_outside}",
    )


def detect_s5(prev: Sequence[int], cur: Sequence[int]) -> Detection | None:
    # Exact signature v0: a [2,8] three-point subset exists in both draws and
    # at least one pair is a uniform translation. Whole-layout equivalence is not required.
    pm = find_partial_gap_matches(prev, [2, 8])
    cm = find_partial_gap_matches(cur, [2, 8])
    carriers: list[tuple[int, ...]] = []
    for p in pm:
        for q in cm:
            ok, _ = translated_subset(p.members, q.members)
            if ok:
                carriers.append(q.members)
    if not carriers:
        return None
    return Detection("S5", _union_members(carriers), "translated partial gap[2,8] subset")


def detect_all(history: Sequence[Sequence[int]], i: int) -> list[Detection]:
    cur = history[i]
    out: list[Detection] = []
    if i >= 1:
        for fn in (detect_s1, detect_s5):
            d = fn(history[i - 1], cur)
            if d:
                out.append(d)
    if i >= 2:
        for fn in (detect_s2, detect_s3, detect_s4):
            d = fn(history[i - 2], history[i - 1], cur)
            if d:
                out.append(d)
    return out


def distance_metrics(carriers: Sequence[int], actual: Sequence[int]) -> dict[str, float]:
    c = sorted(set(int(x) for x in carriers))
    a = sorted(int(x) for x in actual)
    if not c:
        raise ValueError("carrier set must not be empty")
    ds = [min(abs(x - y) for y in c) for x in a]
    return {
        "mean_min_distance": sum(ds) / len(ds),
        "exact_count": float(sum(d == 0 for d in ds)),
        "within1_count": float(sum(d <= 1 for d in ds)),
        "within2_count": float(sum(d <= 2 for d in ds)),
    }


def random_baseline(
    size: int,
    actual: Sequence[int],
    seed: int,
    samples: int = 200,
) -> dict[str, float]:
    rng = random.Random(seed)
    rows = [
        distance_metrics(rng.sample(range(1, 38), size), actual)
        for _ in range(samples)
    ]
    return {
        k: sum(r[k] for r in rows) / len(rows)
        for k in rows[0]
    }


def run(df: pd.DataFrame, random_samples: int = 200) -> dict:
    df = df.sort_values("round").reset_index(drop=True)
    history = [_draw(df.loc[i]) for i in range(len(df))]
    per_structure: dict[str, list[tuple[dict, dict]]] = {f"S{i}": [] for i in range(1, 6)}
    events: list[dict] = []

    # i is the last observed draw used to detect the structure.
    # next draw i+1 is evaluation only and is never used in detection.
    for i in range(2, len(history) - 1):
        actual_next = history[i + 1]
        for d in detect_all(history, i):
            obs = distance_metrics(d.carriers, actual_next)
            base = random_baseline(
                len(d.carriers),
                actual_next,
                seed=int(df.loc[i, "round"]) * 100 + int(d.structure_id[1:]),
                samples=random_samples,
            )
            per_structure[d.structure_id].append((obs, base))
            events.append({
                "structure": d.structure_id,
                "trigger_round": int(df.loc[i, "round"]),
                "next_round": int(df.loc[i + 1, "round"]),
                "carriers": list(d.carriers),
                "detail": d.detail,
                "observed": obs,
                "random_same_size": base,
            })

    def summarize(rows: list[tuple[dict, dict]]) -> dict:
        if not rows:
            return {"events": 0}
        keys = rows[0][0].keys()
        observed = {k: sum(o[k] for o, _ in rows) / len(rows) for k in keys}
        random_mean = {k: sum(b[k] for _, b in rows) / len(rows) for k in keys}
        delta = {
            k: observed[k] - random_mean[k]
            for k in keys
        }
        return {
            "events": len(rows),
            "observed_mean": observed,
            "random_same_size_mean": random_mean,
            "delta_observed_minus_random": delta,
        }

    unconditional_rows = [
        distance_metrics(history[i], history[i + 1])
        for i in range(len(history) - 1)
    ]
    unconditional = {
        k: sum(r[k] for r in unconditional_rows) / len(unconditional_rows)
        for k in unconditional_rows[0]
    }

    return {
        "name": "LOTO7 Astra Observation Model next-distance probe v0",
        "range": {
            "first_round": int(df.iloc[0]["round"]),
            "last_round": int(df.iloc[-1]["round"]),
            "draws": len(df),
        },
        "structures": {k: summarize(v) for k, v in per_structure.items()},
        "unconditional_current_draw_to_next": unconditional,
        "event_count_total": len(events),
        "events": events,
        "boundary": (
            "Exact-signature v0 only. Detection uses current/past draws only; next draw is evaluation only. "
            "Random baseline uses the same carrier-set size. Negative mean_min_distance delta is closer; "
            "positive exact/within deltas are better. This is a descriptive probe, not predictive validation."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/loto7.csv")
    p.add_argument("--random-samples", type=int, default=200)
    p.add_argument("--events-out", default="")
    args = p.parse_args()

    result = run(pd.read_csv(args.csv), random_samples=args.random_samples)
    if args.events_out:
        with open(args.events_out, "w", encoding="utf-8") as f:
            json.dump(result["events"], f, ensure_ascii=False, indent=2)

    compact = dict(result)
    compact.pop("events")
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
