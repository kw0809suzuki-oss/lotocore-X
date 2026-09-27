from __future__ import annotations

from collections import Counter
from pathlib import Path
import math

import numpy as np
import pandas as pd

DATA = Path("data/loto7.csv")
OUT = Path("results/loto7_a_world_box_motion_v0.csv")
NUMBERS = list(range(1, 38))
LONG_WINDOW = 100
RECENT_WINDOW = 20
ALPHA = 0.72
BOX_SIZE = 7
AXES = ("center", "spread", "cluster7")


def draws_from_df(df):
    return [
        [int(row[f"n{i}"]) for i in range(1, 8)]
        for _, row in df.iterrows()
    ]


def frequency_distribution(draws):
    counts = Counter(n for draw in draws for n in draw)
    total = sum(counts.values())
    return {n: counts[n] / total for n in NUMBERS}


def mix_distribution(long_p, recent_p):
    raw = {
        n: ALPHA * long_p[n] + (1.0 - ALPHA) * recent_p[n]
        for n in NUMBERS
    }
    total = sum(raw.values())
    return {n: raw[n] / total for n in NUMBERS}


def box_from_distribution(p):
    ranked = sorted(NUMBERS, key=lambda n: (-p[n], n))
    return tuple(sorted(ranked[:BOX_SIZE]))


def box_distribution(box):
    chosen = set(box)
    return {n: (1.0 / BOX_SIZE if n in chosen else 0.0) for n in NUMBERS}


def state(p):
    center = sum(n * p[n] for n in NUMBERS)
    variance = sum(((n - center) ** 2) * p[n] for n in NUMBERS)
    spread = math.sqrt(variance)
    cluster7 = max(
        sum(p[n] for n in range(start, start + BOX_SIZE))
        for start in range(1, 38 - BOX_SIZE + 1)
    )
    return {"center": center, "spread": spread, "cluster7": cluster7}


def sign(x, eps=1e-12):
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def label(s):
    return {1: "up", 0: "flat", -1: "down"}[s]


def main():
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    draws = draws_from_df(df)

    states = []
    for i in range(LONG_WINDOW, len(draws)):
        long_p = frequency_distribution(draws[i - LONG_WINDOW : i])
        recent_p = frequency_distribution(draws[i - RECENT_WINDOW : i])
        mix_p = mix_distribution(long_p, recent_p)
        box = box_from_distribution(mix_p)
        box_p = box_distribution(box)

        a = state(mix_p)
        b = state(box_p)
        states.append({
            "round": int(df.iloc[i]["round"]),
            "date": df.iloc[i].get("date", ""),
            "box": "-".join(f"{n:02d}" for n in box),
            **{f"a_{k}": v for k, v in a.items()},
            **{f"box_{k}": v for k, v in b.items()},
        })

    s = pd.DataFrame(states)
    rows = []
    for j in range(1, len(s)):
        prev = s.iloc[j - 1]
        cur = s.iloc[j]
        rec = {
            "from_round": int(prev["round"]),
            "to_round": int(cur["round"]),
            "box_from": prev["box"],
            "box_to": cur["box"],
        }
        aligned = 0
        active = 0
        for axis in AXES:
            da = float(cur[f"a_{axis}"] - prev[f"a_{axis}"])
            db = float(cur[f"box_{axis}"] - prev[f"box_{axis}"])
            sa = sign(da)
            sb = sign(db)
            same = sa == sb
            if sa != 0:
                active += 1
                if same:
                    aligned += 1
            rec[f"a_d_{axis}"] = da
            rec[f"box_d_{axis}"] = db
            rec[f"a_dir_{axis}"] = label(sa)
            rec[f"box_dir_{axis}"] = label(sb)
            rec[f"dir_match_{axis}"] = same
        rec["active_a_axes"] = active
        rec["matched_active_axes"] = aligned
        rec["all_active_axes_match"] = (active > 0 and aligned == active)
        rows.append(rec)

    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)

    print("=== LOTO7 A-WORLD BOX MOTION v0 ===")
    print(f"transitions={len(res)} alpha={ALPHA:.2f} box={BOX_SIZE}")
    print("No target-round outcome is used. This compares A-world state motion with box-state motion only.")
    print()

    print("DIRECTION AGREEMENT")
    for axis in AXES:
        m = res[f"dir_match_{axis}"].mean()
        nonflat = res[res[f"a_dir_{axis}"] != "flat"]
        m_active = nonflat[f"dir_match_{axis}"].mean() if len(nonflat) else float("nan")
        print(f"{axis}: all={m:.6f} A-active={m_active:.6f} n_active={len(nonflat)}")

    print()
    print("DELTA CORRELATION")
    for axis in AXES:
        pearson = res[f"a_d_{axis}"].corr(res[f"box_d_{axis}"], method="pearson")
        spearman = res[f"a_d_{axis}"].rank(method="average").corr(res[f"box_d_{axis}"].rank(method="average"), method="pearson")
        print(f"{axis}: pearson={pearson:.6f} spearman={spearman:.6f}")

    print()
    print("MULTI-AXIS")
    print(f"mean matched active axes={res['matched_active_axes'].mean():.6f}")
    print(f"all active axes match={res['all_active_axes_match'].mean():.6f}")
    print("matched_active_axes_dist=" + str(res["matched_active_axes"].value_counts().sort_index().to_dict()))

    print()
    print("RECENT 20 TRANSITIONS")
    cols = [
        "from_round", "to_round",
        "a_d_center", "box_d_center", "dir_match_center",
        "a_d_spread", "box_d_spread", "dir_match_spread",
        "a_d_cluster7", "box_d_cluster7", "dir_match_cluster7",
        "matched_active_axes", "all_active_axes_match",
    ]
    print(res[cols].tail(20).to_string(index=False))
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
