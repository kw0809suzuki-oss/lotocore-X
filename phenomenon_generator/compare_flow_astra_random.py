from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from next_distance_probe import NUM_COLS, detect_all, distance_metrics, random_baseline


@dataclass(frozen=True)
class FlowDetection:
    kind: str
    carriers: tuple[int, ...]
    detail: str


def _draw(row: pd.Series) -> list[int]:
    return sorted(int(row[c]) for c in NUM_COLS)


def detect_flow_return(a: Sequence[int], b: Sequence[int], c: Sequence[int]) -> FlowDetection | None:
    # Existing Flow observation: multiple points disappear for one draw and return.
    sa, sb, sc = set(a), set(b), set(c)
    returned = tuple(sorted((sa & sc) - sb))
    if len(returned) < 2:
        return None
    return FlowDetection("return_after_absence", returned, f"returned={list(returned)}")


def detect_flow_persistence(a: Sequence[int], b: Sequence[int], c: Sequence[int]) -> FlowDetection | None:
    # Existing Flow observations: one point / pair persists while surroundings rewrite.
    common = tuple(sorted(set(a) & set(b) & set(c)))
    if not common or len(common) > 2:
        return None
    # Require substantial surrounding rewrite on both transitions.
    changed_ab = len(set(a) ^ set(b))
    changed_bc = len(set(b) ^ set(c))
    if changed_ab < 6 or changed_bc < 6:
        return None
    return FlowDetection(
        "point_pair_persistence",
        common,
        f"persistent={list(common)}; changed_ab={changed_ab}; changed_bc={changed_bc}",
    )


def detect_flow_width_jump(prev: Sequence[int], cur: Sequence[int]) -> FlowDetection | None:
    # Existing Flow observation: near-preserved total width, zero exact overlap,
    # and a large relocation of the layout centre.
    a, b = sorted(prev), sorted(cur)
    if set(a) & set(b):
        return None
    wa, wb = a[-1] - a[0], b[-1] - b[0]
    if abs(wa - wb) > 1:
        return None
    ca = (a[0] + a[-1]) / 2
    cb = (b[0] + b[-1]) / 2
    if abs(cb - ca) < 10:
        return None
    return FlowDetection(
        "width_preserving_jump",
        tuple(b),
        f"widths={wa},{wb}; center_shift={cb-ca:.1f}",
    )


def detect_flow(history: Sequence[Sequence[int]], i: int) -> list[FlowDetection]:
    out: list[FlowDetection] = []
    if i >= 1:
        d = detect_flow_width_jump(history[i - 1], history[i])
        if d:
            out.append(d)
    if i >= 2:
        for fn in (detect_flow_return, detect_flow_persistence):
            d = fn(history[i - 2], history[i - 1], history[i])
            if d:
                out.append(d)
    return out


def _union_carriers(ds) -> tuple[int, ...]:
    return tuple(sorted({n for d in ds for n in d.carriers}))


def run(df: pd.DataFrame, random_samples: int = 500) -> dict:
    df = df.sort_values("round").reset_index(drop=True)
    history = [_draw(df.loc[i]) for i in range(len(df))]

    buckets = {"flow": [], "astra": []}
    event_rows = []

    for i in range(2, len(history) - 1):
        next_draw = history[i + 1]
        round_no = int(df.loc[i, "round"])

        flow_ds = detect_flow(history, i)
        if flow_ds:
            carriers = _union_carriers(flow_ds)
            obs = distance_metrics(carriers, next_draw)
            base = random_baseline(
                len(carriers), next_draw,
                seed=round_no * 1000 + 11,
                samples=random_samples,
            )
            buckets["flow"].append((obs, base))
            event_rows.append({
                "model": "flow",
                "trigger_round": round_no,
                "next_round": int(df.loc[i + 1, "round"]),
                "carriers": list(carriers),
                "signals": [d.kind for d in flow_ds],
                "observed": obs,
                "random_same_size": base,
            })

        astra_ds = detect_all(history, i)
        if astra_ds:
            carriers = _union_carriers(astra_ds)
            obs = distance_metrics(carriers, next_draw)
            base = random_baseline(
                len(carriers), next_draw,
                seed=round_no * 1000 + 22,
                samples=random_samples,
            )
            buckets["astra"].append((obs, base))
            event_rows.append({
                "model": "astra",
                "trigger_round": round_no,
                "next_round": int(df.loc[i + 1, "round"]),
                "carriers": list(carriers),
                "signals": [d.structure_id for d in astra_ds],
                "observed": obs,
                "random_same_size": base,
            })

    def summarize(rows):
        if not rows:
            return {"events": 0}
        keys = rows[0][0].keys()
        obs = {k: sum(a[k] for a, _ in rows) / len(rows) for k in keys}
        rnd = {k: sum(b[k] for _, b in rows) / len(rows) for k in keys}
        delta = {k: obs[k] - rnd[k] for k in keys}
        return {
            "events": len(rows),
            "observed_mean": obs,
            "random_same_size_mean": rnd,
            "delta_vs_random": delta,
        }

    return {
        "name": "Random vs Flow Observation vs Astra Observation — next-distance run v0",
        "range": {
            "first_round": int(df.iloc[0]["round"]),
            "last_round": int(df.iloc[-1]["round"]),
            "draws": len(df),
        },
        "flow": summarize(buckets["flow"]),
        "astra": summarize(buckets["astra"]),
        "event_rows": event_rows,
        "boundary": (
            "One descriptive run on historical data. Flow uses only already-observed/codified "
            "phenomenon families that can be detected future-blind. Astra uses S1-S5 exact-signature v0. "
            "Random is same carrier-set size per event. Different models may fire on different rounds. "
            "No prediction claim."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/loto7.csv")
    p.add_argument("--random-samples", type=int, default=500)
    args = p.parse_args()
    result = run(pd.read_csv(args.csv), args.random_samples)
    compact = dict(result)
    compact.pop("event_rows")
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
