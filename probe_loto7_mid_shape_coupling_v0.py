from __future__ import annotations

from pathlib import Path
import math
import pandas as pd

DATA = Path("data/loto7.csv")
OUT = Path("results/loto7_mid_shape_coupling_v0.csv")


def draw(row):
    return sorted(int(row[f"n{i}"]) for i in range(1, 8))


def center(xs):
    return sum(xs) / len(xs)


def spread(xs):
    c = center(xs)
    return math.sqrt(sum((x-c)**2 for x in xs) / len(xs))


def mid_count(xs):
    return sum(13 <= n <= 24 for n in xs)


def sign_label(x, eps=1e-12):
    if x > eps:
        return "UP"
    if x < -eps:
        return "DOWN"
    return "FLAT"


def transition(a,b):
    return {
        "mid_delta": mid_count(b) - mid_count(a),
        "center_delta": center(b) - center(a),
        "spread_delta": spread(b) - spread(a),
        "range_delta": (max(b)-min(b)) - (max(a)-min(a)),
    }


def main():
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    draws = [draw(r) for _,r in df.iterrows()]
    ts = [transition(draws[i],draws[i+1]) for i in range(len(draws)-1)]

    rows=[]
    for i in range(len(ts)-1):
        cur = ts[i]
        nxt = ts[i+1]
        rows.append({
            "to_round": int(df.iloc[i+1]["round"]),
            "current_mid_delta": cur["mid_delta"],
            "current_mid_sign": sign_label(cur["mid_delta"]),
            "next_mid_delta": nxt["mid_delta"],
            "next_mid_sign": sign_label(nxt["mid_delta"]),
            "next_center_delta": nxt["center_delta"],
            "next_center_sign": sign_label(nxt["center_delta"]),
            "next_spread_delta": nxt["spread_delta"],
            "next_spread_sign": sign_label(nxt["spread_delta"]),
            "next_range_delta": nxt["range_delta"],
            "next_range_sign": sign_label(nxt["range_delta"]),
            "mid_reversal": (
                (cur["mid_delta"] > 0 and nxt["mid_delta"] < 0) or
                (cur["mid_delta"] < 0 and nxt["mid_delta"] > 0)
            ),
        })

    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT,index=False)

    print("=== LOTO7 MID <-> SHAPE COUPLING v0 ===")
    print("Purpose: observe whether MID-band reversal is coupled to next center/spread/range movement.")
    print("No number identities, prediction rule, thresholds, or fitted weights.")
    print()

    for lab in ["UP","FLAT","DOWN"]:
        part = res[res["current_mid_sign"]==lab]
        print(f"--- CURRENT MID {lab} n={len(part)} ---")
        print(f"mean next_mid_delta={part['next_mid_delta'].mean():+.6f}")
        print(f"mean next_center_delta={part['next_center_delta'].mean():+.6f}")
        print(f"mean next_spread_delta={part['next_spread_delta'].mean():+.6f}")
        print(f"mean next_range_delta={part['next_range_delta'].mean():+.6f}")
        print(f"mid_reversal_rate={part['mid_reversal'].mean():.6%}")
        print()

    print("NEXT SHAPE SIGN DISTRIBUTION BY CURRENT MID SIGN")
    for col in ["next_center_sign","next_spread_sign","next_range_sign"]:
        print(f"--- {col} ---")
        print(pd.crosstab(res["current_mid_sign"],res[col],normalize="index").to_string())
        print()

    print("MID REVERSAL vs NO REVERSAL")
    for label, part in [
        ("REVERSAL",res[res["mid_reversal"]]),
        ("NO_REVERSAL",res[~res["mid_reversal"]]),
    ]:
        print(f"--- {label} n={len(part)} ---")
        print(f"mean next_center_delta={part['next_center_delta'].mean():+.6f}")
        print(f"mean abs next_center_delta={part['next_center_delta'].abs().mean():.6f}")
        print(f"mean next_spread_delta={part['next_spread_delta'].mean():+.6f}")
        print(f"mean abs next_spread_delta={part['next_spread_delta'].abs().mean():.6f}")
        print(f"mean next_range_delta={part['next_range_delta'].mean():+.6f}")
        print(f"mean abs next_range_delta={part['next_range_delta'].abs().mean():.6f}")
        print()

    print("EXACT CURRENT MID DELTA -> NEXT SHAPE MEANS")
    exact = res.groupby("current_mid_delta").agg(
        n=("next_mid_delta","size"),
        next_mid_mean=("next_mid_delta","mean"),
        next_center_mean=("next_center_delta","mean"),
        next_spread_mean=("next_spread_delta","mean"),
        next_range_mean=("next_range_delta","mean"),
        reversal_rate=("mid_reversal","mean"),
    ).sort_index()
    print(exact.to_string())
    print()

    cut=int(len(res)*0.60)
    for tag,part in [("EARLY60",res.iloc[:cut]),("LATE40",res.iloc[cut:])]:
        print(f"--- {tag} ---")
        for lab in ["UP","DOWN"]:
            p=part[part["current_mid_sign"]==lab]
            print(
                f"{lab}: n={len(p)} next_mid={p['next_mid_delta'].mean():+.6f} "
                f"center={p['next_center_delta'].mean():+.6f} "
                f"spread={p['next_spread_delta'].mean():+.6f} "
                f"range={p['next_range_delta'].mean():+.6f} "
                f"reversal={p['mid_reversal'].mean():.6%}"
            )
        print()

    print(f"saved -> {OUT}")


if __name__=="__main__":
    main()
