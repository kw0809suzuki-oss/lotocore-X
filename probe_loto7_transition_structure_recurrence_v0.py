from __future__ import annotations

from pathlib import Path
import math
import statistics

import pandas as pd

DATA = Path("data/loto7.csv")
OUT = Path("results/loto7_transition_structure_recurrence_v0.csv")


def draw(row):
    return sorted(int(row[f"n{i}"]) for i in range(1, 8))


def center(xs):
    return sum(xs) / len(xs)


def spread(xs):
    c = center(xs)
    return math.sqrt(sum((x-c)**2 for x in xs) / len(xs))


def max_gap(xs):
    ys = sorted(xs)
    return max((b-a) for a,b in zip(ys[:-1], ys[1:])) if len(ys) > 1 else 0


def gap_cv(xs):
    ys = sorted(xs)
    gaps = [b-a for a,b in zip(ys[:-1], ys[1:])]
    if not gaps:
        return 0.0
    m = sum(gaps)/len(gaps)
    if m == 0:
        return 0.0
    sd = math.sqrt(sum((g-m)**2 for g in gaps)/len(gaps))
    return sd/m


def band_counts(xs):
    # fixed, non-fitted thirds of 1..37
    return (
        sum(1 <= x <= 12 for x in xs),
        sum(13 <= x <= 24 for x in xs),
        sum(25 <= x <= 37 for x in xs),
    )


def transition_structure(a, b):
    sa, sb = set(a), set(b)
    stay = len(sa & sb)
    enter = len(sb - sa)
    exit_ = len(sa - sb)

    al, am, ah = band_counts(a)
    bl, bm, bh = band_counts(b)

    return {
        "stay_count": stay,
        "enter_count": enter,
        "exit_count": exit_,
        "center_delta": center(b) - center(a),
        "spread_delta": spread(b) - spread(a),
        "range_delta": (max(b)-min(b)) - (max(a)-min(a)),
        "max_gap_delta": max_gap(b) - max_gap(a),
        "gap_cv_delta": gap_cv(b) - gap_cv(a),
        "low_delta": bl - al,
        "mid_delta": bm - am,
        "high_delta": bh - ah,
    }


FEATURES = [
    "stay_count","enter_count","exit_count",
    "center_delta","spread_delta","range_delta",
    "max_gap_delta","gap_cv_delta",
    "low_delta","mid_delta","high_delta",
]

# fixed scales only to put heterogeneous structural quantities on comparable footing.
SCALES = {
    "stay_count": 7.0,
    "enter_count": 7.0,
    "exit_count": 7.0,
    "center_delta": 18.0,
    "spread_delta": 10.0,
    "range_delta": 36.0,
    "max_gap_delta": 36.0,
    "gap_cv_delta": 2.0,
    "low_delta": 7.0,
    "mid_delta": 7.0,
    "high_delta": 7.0,
}


def structure_distance(x, y, features=FEATURES):
    vals=[]
    for f in features:
        vals.append(abs(x[f]-y[f]) / SCALES[f])
    return sum(vals)/len(vals)


def feature_absdiff(x, y, f):
    return abs(x[f]-y[f])


def main():
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    draws = [draw(r) for _,r in df.iterrows()]
    structs = [transition_structure(draws[i], draws[i+1]) for i in range(len(draws)-1)]

    rows=[]

    for i in range(2, len(structs)-1):
        eligible=list(range(0,i-1))
        if not eligible:
            continue

        best_j=min(
            eligible,
            key=lambda j:(structure_distance(structs[i],structs[j]), j)
        )

        cur_dist=structure_distance(structs[i],structs[best_j])
        next_dist=structure_distance(structs[i+1],structs[best_j+1])

        control_next=[
            structure_distance(structs[i+1],structs[k+1])
            for k in eligible
        ]
        control_mean=statistics.mean(control_next)
        lift=control_mean-next_dist  # positive = matched successor structurally closer than generic past successor

        rec={
            "current_to_round":int(df.iloc[i+1]["round"]),
            "matched_to_round":int(df.iloc[best_j+1]["round"]),
            "current_structure_distance":cur_dist,
            "next_structure_distance":next_dist,
            "control_next_distance_mean":control_mean,
            "successor_structure_lift":lift,
        }

        for f in FEATURES:
            rec[f"curdiff_{f}"]=feature_absdiff(structs[i],structs[best_j],f)
            rec[f"nextdiff_{f}"]=feature_absdiff(structs[i+1],structs[best_j+1],f)
            controls=[feature_absdiff(structs[i+1],structs[k+1],f) for k in eligible]
            rec[f"control_nextdiff_{f}"]=statistics.mean(controls)
            rec[f"lift_{f}"]=rec[f"control_nextdiff_{f}"]-rec[f"nextdiff_{f}"]

        rows.append(rec)

    res=pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    res.to_csv(OUT,index=False)

    print("=== LOTO7 TRANSITION STRUCTURE RECURRENCE v0 ===")
    print("Goal: identify which structural features carry the previously observed transition recurrence.")
    print("Number identities are not used in the structure vector.")
    print("No prediction rule, selector, threshold search, or weight fitting.")
    print()
    print(f"evaluated={len(res)}")
    print(f"mean current structure distance={res['current_structure_distance'].mean():.6f}")
    print(f"mean successor structure distance={res['next_structure_distance'].mean():.6f}")
    print(f"mean control successor distance={res['control_next_distance_mean'].mean():.6f}")
    print(f"mean successor structure lift={res['successor_structure_lift'].mean():+.6f}")
    print(f"positive/same/negative={(res['successor_structure_lift']>0).sum()}/{(res['successor_structure_lift']==0).sum()}/{(res['successor_structure_lift']<0).sum()}")
    print()

    print("FEATURE CONTRIBUTION | positive lift means matched successor repeats that feature better than generic control")
    feat_rows=[]
    for f in FEATURES:
        m=res[f"lift_{f}"].mean()
        med=res[f"lift_{f}"].median()
        pos=int((res[f"lift_{f}"]>0).sum())
        neg=int((res[f"lift_{f}"]<0).sum())
        feat_rows.append((m,f,med,pos,neg))
    for m,f,med,pos,neg in sorted(feat_rows, reverse=True):
        print(f"{f}: mean_lift={m:+.6f} median={med:+.6f} pos={pos} neg={neg}")
    print()

    cut=int(len(res)*0.60)
    for label,part in [("EARLY60",res.iloc[:cut]),("LATE40",res.iloc[cut:])]:
        print(f"--- {label} n={len(part)} ---")
        print(f"mean successor structure lift={part['successor_structure_lift'].mean():+.6f}")
        for f in FEATURES:
            print(f"{f}: {part[f'lift_{f}'].mean():+.6f}")
        print()

    print("RECENT 20")
    cols=[
        "current_to_round","matched_to_round",
        "current_structure_distance","next_structure_distance",
        "control_next_distance_mean","successor_structure_lift",
    ]
    print(res[cols].tail(20).to_string(index=False))
    print()
    print(f"saved -> {OUT}")


if __name__=="__main__":
    main()
