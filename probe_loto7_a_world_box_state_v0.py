from __future__ import annotations

from collections import Counter
from pathlib import Path
import math

import pandas as pd

DATA = Path("data/loto7.csv")
OUT = Path("results/loto7_a_world_box_state_v0.csv")
NUMBERS = list(range(1, 38))
LONG_WINDOW = 100
RECENT_WINDOW = 20
ALPHA = 0.72
BOX_SIZE = 7


def draws_from_df(df):
    return [
        [int(row[f"n{i}"]) for i in range(1, 8)]
        for _, row in df.iterrows()
    ]


def frequency_distribution(draws):
    counts = Counter(n for draw in draws for n in draw)
    total = sum(counts.values())
    return {n: counts[n] / total for n in NUMBERS}


def mix_distribution(long_p, recent_p, alpha=ALPHA):
    raw = {
        n: alpha * long_p[n] + (1.0 - alpha) * recent_p[n]
        for n in NUMBERS
    }
    total = sum(raw.values())
    return {n: raw[n] / total for n in NUMBERS}


def box_from_distribution(p):
    ranked = sorted(NUMBERS, key=lambda n: (-p[n], n))
    return tuple(sorted(ranked[:BOX_SIZE]))


def box_distribution(box):
    b = set(box)
    return {n: (1.0 / BOX_SIZE if n in b else 0.0) for n in NUMBERS}


def state(p):
    center = sum(n * p[n] for n in NUMBERS)
    variance = sum(((n - center) ** 2) * p[n] for n in NUMBERS)
    spread = math.sqrt(variance)

    cluster7 = 0.0
    cluster_start = 1
    for start in range(1, 38 - BOX_SIZE + 1):
        mass = sum(p[n] for n in range(start, start + BOX_SIZE))
        if mass > cluster7:
            cluster7 = mass
            cluster_start = start

    return {
        "center": center,
        "spread": spread,
        "cluster7": cluster7,
        "cluster_start": cluster_start,
    }


def between(x, a, b, tol=1e-12):
    lo = min(a, b) - tol
    hi = max(a, b) + tol
    return lo <= x <= hi


def main():
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    draws = draws_from_df(df)
    rows = []

    for i in range(LONG_WINDOW, len(draws)):
        long_p = frequency_distribution(draws[i - LONG_WINDOW : i])
        recent_p = frequency_distribution(draws[i - RECENT_WINDOW : i])
        mix_p = mix_distribution(long_p, recent_p)

        box = box_from_distribution(mix_p)
        box_p = box_distribution(box)

        s_long = state(long_p)
        s_recent = state(recent_p)
        s_mix = state(mix_p)
        s_box = state(box_p)

        corridor_center = between(s_box["center"], s_long["center"], s_recent["center"])
        corridor_spread = between(s_box["spread"], s_long["spread"], s_recent["spread"])
        corridor_cluster = between(s_box["cluster7"], s_long["cluster7"], s_recent["cluster7"])
        corridor_axes = int(corridor_center) + int(corridor_spread) + int(corridor_cluster)

        score_mass = sum(mix_p[n] for n in box)

        rows.append({
            "round": int(df.iloc[i]["round"]),
            "date": df.iloc[i].get("date", ""),
            "box": "-".join(f"{n:02d}" for n in box),
            "long_center": s_long["center"],
            "recent_center": s_recent["center"],
            "mix_center": s_mix["center"],
            "box_center": s_box["center"],
            "center_abs_error": abs(s_box["center"] - s_mix["center"]),
            "center_in_slow_fast_corridor": corridor_center,
            "long_spread": s_long["spread"],
            "recent_spread": s_recent["spread"],
            "mix_spread": s_mix["spread"],
            "box_spread": s_box["spread"],
            "spread_abs_error": abs(s_box["spread"] - s_mix["spread"]),
            "spread_in_slow_fast_corridor": corridor_spread,
            "long_cluster7": s_long["cluster7"],
            "recent_cluster7": s_recent["cluster7"],
            "mix_cluster7": s_mix["cluster7"],
            "box_cluster7": s_box["cluster7"],
            "cluster_abs_error": abs(s_box["cluster7"] - s_mix["cluster7"]),
            "cluster_in_slow_fast_corridor": corridor_cluster,
            "corridor_axes": corridor_axes,
            "mix_mass_retained_by_box": score_mass,
        })

    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)

    print("=== LOTO7 A-WORLD BOX STATE v0 ===")
    print(f"rounds={len(res)} long={LONG_WINDOW} recent={RECENT_WINDOW} alpha={ALPHA:.2f} box={BOX_SIZE}")
    print("No future outcomes are used. This probe only compares observed A-world state with its 7-number compression.")
    print()
    print("MEAN STATE")
    for axis in ("center", "spread", "cluster7"):
        print(
            f"{axis}: "
            f"long={res[f'long_{axis}'].mean():.6f} "
            f"recent={res[f'recent_{axis}'].mean():.6f} "
            f"mix={res[f'mix_{axis}'].mean():.6f} "
            f"box={res[f'box_{axis}'].mean():.6f} "
            f"abs_error={res[f'{axis.split('7')[0]}_abs_error' if axis == 'cluster7' else f'{axis}_abs_error'].mean():.6f}"
        )

    print()
    print("SLOW-FAST CORRIDOR")
    print(f"center in corridor={res['center_in_slow_fast_corridor'].mean():.6f}")
    print(f"spread in corridor={res['spread_in_slow_fast_corridor'].mean():.6f}")
    print(f"cluster in corridor={res['cluster_in_slow_fast_corridor'].mean():.6f}")
    print(f"all 3 axes in corridor={(res['corridor_axes'] == 3).mean():.6f}")
    print(f"mean corridor axes={res['corridor_axes'].mean():.6f} / 3")

    print()
    print("COMPRESSION")
    print(f"mean mix mass retained by 7-number box={res['mix_mass_retained_by_box'].mean():.6f}")
    print(f"min mix mass retained={res['mix_mass_retained_by_box'].min():.6f}")
    print(f"max mix mass retained={res['mix_mass_retained_by_box'].max():.6f}")

    print()
    print("LAST 10")
    cols = [
        "round", "box",
        "mix_center", "box_center", "center_abs_error",
        "mix_spread", "box_spread", "spread_abs_error",
        "mix_cluster7", "box_cluster7", "cluster_abs_error",
        "corridor_axes", "mix_mass_retained_by_box",
    ]
    print(res[cols].tail(10).to_string(index=False))
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
