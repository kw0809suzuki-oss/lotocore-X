from __future__ import annotations

import argparse
import json
import random
from typing import Sequence

import pandas as pd

from compare_flow_astra_random import detect_flow
from compare_random_vs_astra_play import evaluate_set, random_ticket
from next_distance_probe import NUM_COLS, detect_all


def _draw(row: pd.Series) -> list[int]:
    return sorted(int(row[c]) for c in NUM_COLS)


def _ticket_from_carriers(
    carriers: Sequence[int],
    rng: random.Random,
) -> list[int]:
    fixed = sorted(set(int(x) for x in carriers if 1 <= int(x) <= 37))
    if len(fixed) > 7:
        fixed = fixed[:7]
    pool = [n for n in range(1, 38) if n not in fixed]
    rest = rng.sample(pool, 7 - len(fixed))
    return sorted(fixed + rest)


def _model20(detections, seed: int, round_no: int, side_code: int) -> list[list[int]]:
    tickets = []
    for j in range(20):
        d = detections[j % len(detections)]
        rng = random.Random(seed * 1_000_000 + round_no * 100 + side_code * 20 + j)
        tickets.append(_ticket_from_carriers(d.carriers, rng))
    return tickets


def run(df: pd.DataFrame, seed: int = 20261004) -> dict:
    df = df.sort_values("round").reset_index(drop=True)
    history = [_draw(df.loc[i]) for i in range(len(df))]
    rows = []

    for i in range(2, len(history) - 1):
        flow_ds = detect_flow(history, i)
        astra_ds = detect_all(history, i)
        if not flow_ds or not astra_ds:
            continue

        trigger_round = int(df.loc[i, "round"])
        actual = history[i + 1]
        actual_round = int(df.loc[i + 1, "round"])

        rrng = random.Random(seed * 1_000_000 + trigger_round * 100)
        random20 = [random_ticket(rrng) for _ in range(20)]
        flow20 = _model20(flow_ds, seed, trigger_round, 1)
        astra20 = _model20(astra_ds, seed, trigger_round, 2)

        rows.append({
            "trigger_round": trigger_round,
            "actual_round": actual_round,
            "actual": actual,
            "flow_signals": [d.kind for d in flow_ds],
            "astra_signals": [d.structure_id for d in astra_ds],
            "random": evaluate_set(random20, actual),
            "flow": evaluate_set(flow20, actual),
            "astra": evaluate_set(astra20, actual),
            "sample_tickets": {
                "random": random20[:3],
                "flow": flow20[:3],
                "astra": astra20[:3],
            },
        })

    def agg(side: str) -> dict:
        vals = [r[side] for r in rows]
        if not vals:
            return {"rounds": 0}
        return {
            "rounds": len(vals),
            "mean_max_hit": sum(v["max_hit"] for v in vals) / len(vals),
            "max_hit_4plus_rounds": sum(v["max_hit"] >= 4 for v in vals),
            "ticket_3plus_total": sum(v["tickets_3plus"] for v in vals),
            "ticket_4plus_total": sum(v["tickets_4plus"] for v in vals),
            "ticket_5plus_total": sum(v["tickets_5plus"] for v in vals),
        }

    random_agg = agg("random")
    flow_agg = agg("flow")
    astra_agg = agg("astra")

    def diff(a: dict, b: dict) -> dict:
        keys = [
            "mean_max_hit",
            "max_hit_4plus_rounds",
            "ticket_3plus_total",
            "ticket_4plus_total",
            "ticket_5plus_total",
        ]
        return {k: a[k] - b[k] for k in keys}

    return {
        "name": "Random20 vs Flow20 vs Astra20 external next-draw comparison v0",
        "window": {
            "first_history_round": int(df.iloc[0]["round"]),
            "last_history_round": int(df.iloc[-1]["round"]),
            "common_fire_rounds": len(rows),
        },
        "random20": random_agg,
        "flow20": flow_agg,
        "astra20": astra_agg,
        "flow_minus_random": diff(flow_agg, random_agg) if rows else {},
        "astra_minus_random": diff(astra_agg, random_agg) if rows else {},
        "flow_minus_astra": diff(flow_agg, astra_agg) if rows else {},
        "round_rows": rows,
        "boundary": (
            "External historical replay. At each trigger round, only information available up to that round "
            "is used. Evaluation opens the next actual draw afterward. Only rounds where both Flow and Astra "
            "fire are included. Each side outputs 20 concrete seven-number tickets. Model tickets preserve "
            "their detected observation carriers and fill remaining slots randomly. This is one play run, "
            "not predictive validation."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/loto7.csv")
    p.add_argument("--seed", type=int, default=20261004)
    p.add_argument("--rows-out", default="")
    args = p.parse_args()

    result = run(pd.read_csv(args.csv), args.seed)
    if args.rows_out:
        with open(args.rows_out, "w", encoding="utf-8") as f:
            json.dump(result["round_rows"], f, ensure_ascii=False, indent=2)
    compact = dict(result)
    compact.pop("round_rows")
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
