from __future__ import annotations

from pathlib import Path
import statistics

import pandas as pd

DATA = Path("data/loto7.csv")
OUT = Path("results/loto7_band_flow_structure_v0.csv")


def draw(row):
    return sorted(int(row[f"n{i}"]) for i in range(1, 8))


def band(n):
    if 1 <= n <= 12:
        return "LOW"
    if 13 <= n <= 24:
        return "MID"
    return "HIGH"


BANDS = ("LOW","MID","HIGH")


def band_counts(xs):
    c={b:0 for b in BANDS}
    for n in xs:
        c[band(n)] += 1
    return c


def transition_band_structure(a,b):
    ca,cb=band_counts(a),band_counts(b)
    delta={k:cb[k]-ca[k] for k in BANDS}

    # Flow matrix is defined only for numbers that leave / enter the draw.
    # It preserves band-level mass movement without using number identity.
    exits=[band(n) for n in set(a)-set(b)]
    enters=[band(n) for n in set(b)-set(a)]

    # We do not invent pairings between individual exits and enters.
    # Instead retain source and destination marginals.
    out_by={k:exits.count(k) for k in BANDS}
    in_by={k:enters.count(k) for k in BANDS}

    return {
        "low_delta":delta["LOW"],
        "mid_delta":delta["MID"],
        "high_delta":delta["HIGH"],
        "low_out":out_by["LOW"],"mid_out":out_by["MID"],"high_out":out_by["HIGH"],
        "low_in":in_by["LOW"],"mid_in":in_by["MID"],"high_in":in_by["HIGH"],
    }


GROUPS={
    "LOW_ONLY":["low_delta"],
    "MID_ONLY":["mid_delta"],
    "HIGH_ONLY":["high_delta"],
    "DELTA_VECTOR":["low_delta","mid_delta","high_delta"],
    "OUTFLOW":["low_out","mid_out","high_out"],
    "INFLOW":["low_in","mid_in","high_in"],
    "FLOW_MARGINALS":["low_out","mid_out","high_out","low_in","mid_in","high_in"],
    "DELTA+FLOW":["low_delta","mid_delta","high_delta","low_out","mid_out","high_out","low_in","mid_in","high_in"],
}

SCALES={k:7.0 for k in {
    "low_delta","mid_delta","high_delta",
    "low_out","mid_out","high_out",
    "low_in","mid_in","high_in",
}}


def dist(x,y,features):
    return sum(abs(x[f]-y[f])/SCALES[f] for f in features)/len(features)


def evaluate(structs,features):
    rows=[]
    for i in range(2,len(structs)-1):
        eligible=list(range(0,i-1))
        j=min(eligible,key=lambda k:(dist(structs[i],structs[k],features),k))
        next_d=dist(structs[i+1],structs[j+1],features)
        controls=[dist(structs[i+1],structs[k+1],features) for k in eligible]
        control=statistics.mean(controls)
        rows.append((control-next_d,dist(structs[i],structs[j],features),next_d,control))
    return rows


def main():
    df=pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    draws=[draw(r) for _,r in df.iterrows()]
    structs=[transition_band_structure(draws[i],draws[i+1]) for i in range(len(draws)-1)]

    out=[]
    print("=== LOTO7 BAND FLOW STRUCTURE v0 ===")
    print("Purpose: identify whether BAND recurrence comes from a particular region or from source/destination flow structure.")
    print("No number identity, no prediction rule, no fitted thresholds or weights.")
    print()

    for name,features in GROUPS.items():
        rows=evaluate(structs,features)
        lifts=[r[0] for r in rows]
        n=len(rows); cut=int(n*0.60)
        rec={
            "group":name,
            "features":"|".join(features),
            "n":n,
            "mean_lift":statistics.mean(lifts),
            "median_lift":statistics.median(lifts),
            "positive":sum(x>0 for x in lifts),
            "same":sum(x==0 for x in lifts),
            "negative":sum(x<0 for x in lifts),
            "early60_mean_lift":statistics.mean(lifts[:cut]),
            "late40_mean_lift":statistics.mean(lifts[cut:]),
        }
        out.append(rec)
        print(f"--- {name} ---")
        print(f"features={','.join(features)}")
        print(f"mean_lift={rec['mean_lift']:+.6f} median={rec['median_lift']:+.6f}")
        print(f"positive/same/negative={rec['positive']}/{rec['same']}/{rec['negative']}")
        print(f"early60={rec['early60_mean_lift']:+.6f} late40={rec['late40_mean_lift']:+.6f}")
        print()

    res=pd.DataFrame(out)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    res.to_csv(OUT,index=False)

    print("ORDER BY MEAN LIFT")
    print(res.sort_values("mean_lift",ascending=False)[[
        "group","mean_lift","median_lift","positive","negative",
        "early60_mean_lift","late40_mean_lift"
    ]].to_string(index=False))
    print()
    print(f"saved -> {OUT}")


if __name__=="__main__":
    main()
