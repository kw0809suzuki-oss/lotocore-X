from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from typing import Sequence

import pandas as pd

from compare_flow_astra_random import detect_flow
from compare_random_vs_astra_play import evaluate_set, random_ticket
from external_threeway_play import _ticket_from_carriers
from next_distance_probe import NUM_COLS, detect_all

NUMS = list(range(1, 38))
PAIR_INDEX = {}
_k = 0
for a in NUMS:
    for b in range(a + 1, 38):
        PAIR_INDEX[(a, b)] = _k
        _k += 1
NP = _k


def _draw(row: pd.Series) -> list[int]:
    return sorted(int(row[c]) for c in NUM_COLS)


def _model30(detections, seed: int, round_no: int, side_code: int) -> list[list[int]]:
    out = []
    seen = set()
    j = 0
    attempts = 0
    while len(out) < 30 and attempts < 10000:
        attempts += 1
        d = detections[j % len(detections)]
        rng = random.Random(seed * 1_000_000 + round_no * 10_000 + side_code * 100 + attempts)
        t = tuple(_ticket_from_carriers(d.carriers, rng))
        j += 1
        if t in seen:
            continue
        seen.add(t)
        out.append(list(t))
    if len(out) != 30:
        raise RuntimeError("could not build 30 unique model tickets")
    return out


def ticket_pairs(t):
    return [PAIR_INDEX[(t[i], t[j])] for i in range(7) for j in range(i + 1, 7)]


def select20(pool30, full60_counts, target, global_sel_counts, global_pair_counts):
    rem = list(enumerate(pool30))
    out = []
    pairlists = [ticket_pairs(t) for t in pool30]
    while len(out) < 20:
        best = None
        for idx, t in rem:
            usage_ex = 0.0
            usage_delta = 0.0
            for n in t:
                before = global_sel_counts[n]
                after = before + 1
                usage_ex += max(0.0, after - target[n])
                usage_delta += abs(after - target[n]) - abs(before - target[n])
            pair_pen = sum(global_pair_counts[p] for p in pairlists[idx])
            score = (usage_ex, pair_pen, usage_delta, idx)
            if best is None or score < best[0]:
                best = (score, idx, t)
        _, idx, t = best
        out.append(t)
        rem.remove((idx, t))
        for n in t:
            global_sel_counts[n] += 1
        for p in pairlists[idx]:
            global_pair_counts[p] += 1
    return out


def assembler40(a30, b30):
    full = a30 + b30
    counts = Counter(n for t in full for n in t)
    target = {n: counts[n] * 2 / 3 for n in NUMS}
    sc = Counter()
    pc = [0] * NP
    sig = sum(sum(t) for t in full)
    if sig % 2 == 0:
        a = select20(a30, counts, target, sc, pc)
        b = select20(b30, counts, target, sc, pc)
    else:
        b = select20(b30, counts, target, sc, pc)
        a = select20(a30, counts, target, sc, pc)
    return a + b


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

        rrng = random.Random(seed * 1_000_000 + trigger_round * 100)
        random40 = [random_ticket(rrng) for _ in range(40)]

        flow_a30 = _model30(flow_ds, seed, trigger_round, 11)
        flow_b30 = _model30(flow_ds, seed, trigger_round, 12)
        astra_a30 = _model30(astra_ds, seed, trigger_round, 21)
        astra_b30 = _model30(astra_ds, seed, trigger_round, 22)

        flow40 = assembler40(flow_a30, flow_b30)
        astra40 = assembler40(astra_a30, astra_b30)

        rows.append({
            "trigger_round": trigger_round,
            "actual_round": int(df.loc[i + 1, "round"]),
            "actual": actual,
            "random": evaluate_set(random40, actual),
            "flow": evaluate_set(flow40, actual),
            "astra": evaluate_set(astra40, actual),
        })

    def agg(side):
        vals = [r[side] for r in rows]
        return {
            "rounds": len(vals),
            "mean_max_hit": sum(v["max_hit"] for v in vals) / len(vals),
            "max_hit_4plus_rounds": sum(v["max_hit"] >= 4 for v in vals),
            "ticket_3plus_total": sum(v["tickets_3plus"] for v in vals),
            "ticket_4plus_total": sum(v["tickets_4plus"] for v in vals),
            "ticket_5plus_total": sum(v["tickets_5plus"] for v in vals),
        }

    ra, fl, aa = agg("random"), agg("flow"), agg("astra")
    keys = ["mean_max_hit", "max_hit_4plus_rounds", "ticket_3plus_total", "ticket_4plus_total", "ticket_5plus_total"]
    def diff(x, y):
        return {k: x[k] - y[k] for k in keys}

    return {
        "name": "Random40 vs Flow60→Assembler40 vs Astra60→Assembler40 external comparison",
        "common_fire_rounds": len(rows),
        "random40": ra,
        "flow60_to40": fl,
        "astra60_to40": aa,
        "flow_minus_random": diff(fl, ra),
        "astra_minus_random": diff(aa, ra),
        "flow_minus_astra": diff(fl, aa),
        "boundary": (
            "Historical external replay. Flow and Astra each generate two independent 30-ticket modules "
            "from observations available at the trigger round; the recovered Assembler40 rule selects "
            "20 from each module using scaled number-use preservation and pair-reuse suppression. "
            "The next actual draw is revealed only for evaluation."
        ),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/loto7.csv")
    p.add_argument("--seed", type=int, default=20261004)
    args = p.parse_args()
    print(json.dumps(run(pd.read_csv(args.csv), args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
