from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass

import pandas as pd


NUM_COLS = [f"n{i}" for i in range(1, 8)]


def shortest_gap_pair(draw: list[int]) -> tuple[int, int, int]:
    xs = sorted(draw)
    pairs = [(b - a, a, b) for a, b in zip(xs, xs[1:])]
    gap, a, b = min(pairs)
    return a, b, gap


def random_ticket(rng: random.Random) -> list[int]:
    return sorted(rng.sample(range(1, 38), 7))


def astra_ticket(rng: random.Random, prev_draw: list[int]) -> list[int]:
    _, _, gap = shortest_gap_pair(prev_draw)
    starts = [s for s in range(1, 38 - gap) if s + gap <= 37]
    start = rng.choice(starts)
    a, b = start, start + gap
    pool = [n for n in range(1, 38) if n not in {a, b}]
    rest = rng.sample(pool, 5)
    return sorted([a, b, *rest])


def hits(ticket: list[int], actual: list[int]) -> int:
    return len(set(ticket) & set(actual))


def evaluate_set(tickets: list[list[int]], actual: list[int]) -> dict:
    hs = [hits(t, actual) for t in tickets]
    return {
        "max_hit": max(hs),
        "tickets_3plus": sum(h >= 3 for h in hs),
        "tickets_4plus": sum(h >= 4 for h in hs),
        "tickets_5plus": sum(h >= 5 for h in hs),
    }


def run(df: pd.DataFrame, last_n: int = 100, seed: int = 20261003) -> dict:
    df = df.sort_values("round").reset_index(drop=True)
    rows = []
    start = max(1, len(df) - last_n)
    for i in range(start, len(df)):
        prev_draw = [int(df.loc[i - 1, c]) for c in NUM_COLS]
        actual = [int(df.loc[i, c]) for c in NUM_COLS]
        round_no = int(df.loc[i, "round"])

        rrng = random.Random(seed * 100000 + round_no)
        arng = random.Random(seed * 100000 + round_no)

        random20 = [random_ticket(rrng) for _ in range(20)]
        astra20 = [astra_ticket(arng, prev_draw) for _ in range(20)]

        r = evaluate_set(random20, actual)
        a = evaluate_set(astra20, actual)
        rows.append({"round": round_no, "random": r, "astra": a})

    def agg(side: str) -> dict:
        vals = [x[side] for x in rows]
        return {
            "rounds": len(vals),
            "mean_max_hit": sum(v["max_hit"] for v in vals) / len(vals),
            "max_hit_4plus_rounds": sum(v["max_hit"] >= 4 for v in vals),
            "ticket_3plus_total": sum(v["tickets_3plus"] for v in vals),
            "ticket_4plus_total": sum(v["tickets_4plus"] for v in vals),
            "ticket_5plus_total": sum(v["tickets_5plus"] for v in vals),
        }

    ra, aa = agg("random"), agg("astra")
    return {
        "name": "Random20 vs Astra20 simple play",
        "rule": "Astra20 preserves one shortest-gap two-point relation from previous draw; remaining five points random.",
        "window": {
            "first_round": rows[0]["round"],
            "last_round": rows[-1]["round"],
            "rounds": len(rows),
        },
        "random20": ra,
        "astra20": aa,
        "difference_astra_minus_random": {
            k: aa[k] - ra[k]
            for k in [
                "mean_max_hit",
                "max_hit_4plus_rounds",
                "ticket_3plus_total",
                "ticket_4plus_total",
                "ticket_5plus_total",
            ]
        },
        "boundary": "Play comparison only. This is not a validated Astra selector or prediction claim.",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/loto7.csv")
    p.add_argument("--last-n", type=int, default=100)
    p.add_argument("--seed", type=int, default=20261003)
    args = p.parse_args()
    df = pd.read_csv(args.csv)
    print(json.dumps(run(df, args.last_n, args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
