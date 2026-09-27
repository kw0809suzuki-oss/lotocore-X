from __future__ import annotations

from pathlib import Path
import random
import statistics

import pandas as pd

import lotocore

DATA = Path("data/loto7.csv")
WINDOW = 100
SHUFFLES = 500
SEED = 20260927
NUMBERS = list(range(1, 38))


def sgn(x, eps=1e-12):
    return 1 if x > eps else (-1 if x < -eps else 0)


def draw_tuple(row):
    return tuple(sorted(int(row[f"n{i}"]) for i in range(1, 8)))


def center(xs):
    return sum(xs) / 7.0


def variance(xs):
    c = center(xs)
    return sum((x-c)**2 for x in xs) / 7.0


def largest_gap_mid(xs):
    gaps = [(xs[i+1]-xs[i], (xs[i+1]+xs[i])/2.0, i) for i in range(6)]
    gaps.sort(key=lambda z: (-z[0], z[2]))
    return gaps[0][1]


def box(history):
    core = tuple(sorted(lotocore.predict(history).numbers))
    snap = lotocore.score_snapshot(history)
    ranks = {int(k): int(v) for k, v in snap["ranks"].items()}
    ranked = sorted(NUMBERS, key=lambda n: (ranks[n], n))
    boundary = tuple(n for n in ranked if n not in set(core))[:6]
    return {
        "core": core,
        "boundary": boundary,
        "center": center(core),
        "spread": variance(core),
        "gap_mid": largest_gap_mid(core),
    }


def transition(a, b):
    return {
        "center_dir": sgn(b["center"] - a["center"]),
        "spread_dir": sgn(b["spread"] - a["spread"]),
        "turnover": 7 - len(set(a["core"]) & set(b["core"])),
        "boundary_in": len(set(b["core"]) & set(a["boundary"])),
        "gap_dir": sgn(b["gap_mid"] - a["gap_mid"]),
    }


def actual_transition(a, b):
    return {
        "center_dir": sgn(center(b) - center(a)),
        "spread_dir": sgn(variance(b) - variance(a)),
        "gap_dir": sgn(largest_gap_mid(b) - largest_gap_mid(a)),
    }


def sign_agree(x, y):
    return 1.0 if x == y else 0.0


def trans_similarity(a, b):
    # Equal-weight toy similarity. Turnover/boundary are softly compared.
    sign_score = (
        sign_agree(a["center_dir"], b["center_dir"])
        + sign_agree(a["spread_dir"], b["spread_dir"])
        + sign_agree(a["gap_dir"], b["gap_dir"])
    ) / 3.0
    turnover_score = 1.0 - min(abs(a["turnover"] - b["turnover"]), 7) / 7.0
    boundary_score = 1.0 - min(abs(a["boundary_in"] - b["boundary_in"]), 6) / 6.0
    return (3*sign_score + turnover_score + boundary_score) / 5.0


def mean(xs):
    return sum(xs)/len(xs) if xs else float("nan")


def main():
    rng = random.Random(SEED)
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)

    # Box_r = model state built only from draws before round r.
    boxes = {}
    for i in range(WINDOW, len(df)):
        r = int(df.iloc[i]["round"])
        boxes[r] = box(df.iloc[i-WINDOW:i])

    records = []
    for i in range(WINDOW+1, len(df)):
        r = int(df.iloc[i]["round"])
        prev_r = int(df.iloc[i-1]["round"])
        if prev_r not in boxes or r not in boxes:
            continue

        bt = transition(boxes[prev_r], boxes[r])
        at = actual_transition(draw_tuple(df.iloc[i-1]), draw_tuple(df.iloc[i]))

        records.append({
            "round": r,
            "box_transition": bt,
            "actual_transition": at,
        })

    # 1) Does box motion have temporal continuity?
    seq_sim = [
        trans_similarity(records[i-1]["box_transition"], records[i]["box_transition"])
        for i in range(1, len(records))
    ]
    observed_seq = mean(seq_sim)

    trans_list = [r["box_transition"] for r in records]
    null_seq = []
    for _ in range(SHUFFLES):
        p = trans_list[:]
        rng.shuffle(p)
        null_seq.append(mean(trans_similarity(p[i-1], p[i]) for i in range(1, len(p))))

    # 2) Does pre-target box motion align with realized draw motion?
    obs_center = mean(sign_agree(r["box_transition"]["center_dir"], r["actual_transition"]["center_dir"]) for r in records)
    obs_spread = mean(sign_agree(r["box_transition"]["spread_dir"], r["actual_transition"]["spread_dir"]) for r in records)
    obs_gap = mean(sign_agree(r["box_transition"]["gap_dir"], r["actual_transition"]["gap_dir"]) for r in records)
    obs_three = mean([
        mean([
            sign_agree(r["box_transition"]["center_dir"], r["actual_transition"]["center_dir"]),
            sign_agree(r["box_transition"]["spread_dir"], r["actual_transition"]["spread_dir"]),
            sign_agree(r["box_transition"]["gap_dir"], r["actual_transition"]["gap_dir"]),
        ]) for r in records
    ])

    actuals = [r["actual_transition"] for r in records]
    null_center, null_spread, null_gap, null_three = [], [], [], []
    for _ in range(SHUFFLES):
        p = actuals[:]
        rng.shuffle(p)
        null_center.append(mean(sign_agree(records[i]["box_transition"]["center_dir"], p[i]["center_dir"]) for i in range(len(records))))
        null_spread.append(mean(sign_agree(records[i]["box_transition"]["spread_dir"], p[i]["spread_dir"]) for i in range(len(records))))
        null_gap.append(mean(sign_agree(records[i]["box_transition"]["gap_dir"], p[i]["gap_dir"]) for i in range(len(records))))
        null_three.append(mean(
            mean([
                sign_agree(records[i]["box_transition"]["center_dir"], p[i]["center_dir"]),
                sign_agree(records[i]["box_transition"]["spread_dir"], p[i]["spread_dir"]),
                sign_agree(records[i]["box_transition"]["gap_dir"], p[i]["gap_dir"]),
            ]) for i in range(len(records))
        ))

    def p_ge(null, obs):
        return sum(x >= obs for x in null) / len(null)

    print("=== LOTO7 BOX TIME-SHIFT TOY v0 ===")
    print(f"transitions={len(records)} rounds={records[0]['round']}..{records[-1]['round']}")
    print("Box(t) is built only from draws before target round t.")
    print("Motion features: center direction / spread direction / core turnover / boundary->core / largest-gap movement.")
    print()
    print("1) BOX MOTION CONTINUITY")
    print(f"observed consecutive similarity={observed_seq:.6f}")
    print(f"shuffle mean={statistics.mean(null_seq):.6f} advantage={observed_seq-statistics.mean(null_seq):+.6f}")
    print(f"shuffle p(>=observed)={p_ge(null_seq, observed_seq):.6f}")
    print()
    print("2) BOX MOTION vs REALIZED DRAW MOTION")
    print(f"center direction agreement: observed={obs_center:.6f} shuffle={statistics.mean(null_center):.6f} advantage={obs_center-statistics.mean(null_center):+.6f} p={p_ge(null_center, obs_center):.6f}")
    print(f"spread direction agreement: observed={obs_spread:.6f} shuffle={statistics.mean(null_spread):.6f} advantage={obs_spread-statistics.mean(null_spread):+.6f} p={p_ge(null_spread, obs_spread):.6f}")
    print(f"gap direction agreement:    observed={obs_gap:.6f} shuffle={statistics.mean(null_gap):.6f} advantage={obs_gap-statistics.mean(null_gap):+.6f} p={p_ge(null_gap, obs_gap):.6f}")
    print(f"3-feature mean agreement:   observed={obs_three:.6f} shuffle={statistics.mean(null_three):.6f} advantage={obs_three-statistics.mean(null_three):+.6f} p={p_ge(null_three, obs_three):.6f}")
    print()
    print("RECENT 15")
    for rec in records[-15:]:
        bt, at = rec["box_transition"], rec["actual_transition"]
        print(
            f"r{rec['round']}: "
            f"BOX center={bt['center_dir']:+d} spread={bt['spread_dir']:+d} "
            f"turnover={bt['turnover']} b->c={bt['boundary_in']} gap={bt['gap_dir']:+d} | "
            f"ACT center={at['center_dir']:+d} spread={at['spread_dir']:+d} gap={at['gap_dir']:+d}"
        )


if __name__ == "__main__":
    main()
