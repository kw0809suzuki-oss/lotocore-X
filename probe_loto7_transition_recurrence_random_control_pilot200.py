from __future__ import annotations

from pathlib import Path
import math
import random
import statistics

import pandas as pd

DATA = Path("data/loto7.csv")
OUT = Path("results/loto7_transition_recurrence_random_control_v0_pilot200.csv")
SIMS = 200
SEED = 20260929


def draw(row):
    return frozenset(int(row[f"n{i}"]) for i in range(1, 8))


def transition(a,b):
    return {
        "stay": frozenset(a & b),
        "exit": frozenset(a - b),
        "enter": frozenset(b - a),
    }


def jaccard(a,b):
    if not a and not b:
        return 1.0
    return len(a & b)/len(a | b)


def transition_similarity(x,y):
    return (
        jaccard(x["stay"],y["stay"])
        + jaccard(x["exit"],y["exit"])
        + jaccard(x["enter"],y["enter"])
    )/3.0


def center(xs):
    xs=list(xs)
    return sum(xs)/len(xs)


def spread(xs):
    xs=list(xs)
    c=center(xs)
    return math.sqrt(sum((x-c)**2 for x in xs)/len(xs))


def max_gap(xs):
    ys=sorted(xs)
    return max((b-a) for a,b in zip(ys[:-1],ys[1:])) if len(ys)>1 else 0


def gap_cv(xs):
    ys=sorted(xs)
    gaps=[b-a for a,b in zip(ys[:-1],ys[1:])]
    if not gaps:
        return 0.0
    m=sum(gaps)/len(gaps)
    if m==0:
        return 0.0
    sd=math.sqrt(sum((g-m)**2 for g in gaps)/len(gaps))
    return sd/m


def band_counts(xs):
    return (
        sum(1 <= x <= 12 for x in xs),
        sum(13 <= x <= 24 for x in xs),
        sum(25 <= x <= 37 for x in xs),
    )


def structural_transition(a,b):
    sa,sb=set(a),set(b)
    al,am,ah=band_counts(a)
    bl,bm,bh=band_counts(b)
    return {
        "stay_count":len(sa & sb),
        "enter_count":len(sb-sa),
        "exit_count":len(sa-sb),
        "center_delta":center(b)-center(a),
        "spread_delta":spread(b)-spread(a),
        "range_delta":(max(b)-min(b))-(max(a)-min(a)),
        "max_gap_delta":max_gap(b)-max_gap(a),
        "gap_cv_delta":gap_cv(b)-gap_cv(a),
        "low_delta":bl-al,
        "mid_delta":bm-am,
        "high_delta":bh-ah,
    }


FEATURES=[
    "stay_count","enter_count","exit_count",
    "center_delta","spread_delta","range_delta","max_gap_delta","gap_cv_delta",
    "low_delta","mid_delta","high_delta",
]
SCALES={
    "stay_count":7.0,"enter_count":7.0,"exit_count":7.0,
    "center_delta":18.0,"spread_delta":10.0,"range_delta":36.0,
    "max_gap_delta":36.0,"gap_cv_delta":2.0,
    "low_delta":7.0,"mid_delta":7.0,"high_delta":7.0,
}


def structure_distance(x,y):
    return sum(abs(x[f]-y[f])/SCALES[f] for f in FEATURES)/len(FEATURES)


def whole_recurrence_lift(draws):
    ts=[transition(draws[i],draws[i+1]) for i in range(len(draws)-1)]
    lifts=[]
    for i in range(2,len(ts)-1):
        eligible=list(range(0,i-1))
        sims=[(transition_similarity(ts[i],ts[j]),j) for j in eligible]
        _,j=max(sims,key=lambda x:(x[0],-x[1]))
        obs=transition_similarity(ts[i+1],ts[j+1])
        ctrl=statistics.mean(transition_similarity(ts[i+1],ts[k+1]) for k in eligible)
        lifts.append(obs-ctrl)
    return statistics.mean(lifts)


def structure_recurrence_lift(draws):
    ts=[structural_transition(draws[i],draws[i+1]) for i in range(len(draws)-1)]
    lifts=[]
    for i in range(2,len(ts)-1):
        eligible=list(range(0,i-1))
        j=min(eligible,key=lambda k:(structure_distance(ts[i],ts[k]),k))
        obs=structure_distance(ts[i+1],ts[j+1])
        ctrl=statistics.mean(structure_distance(ts[i+1],ts[k+1]) for k in eligible)
        lifts.append(ctrl-obs)
    return statistics.mean(lifts)


def empirical_summary(vals,obs):
    mu=statistics.mean(vals)
    sd=statistics.pstdev(vals)
    z=(obs-mu)/sd if sd else float("nan")
    pct=sum(v <= obs for v in vals)/len(vals)
    p2=sum(abs(v-mu)>=abs(obs-mu) for v in vals)/len(vals)
    return mu,sd,z,pct,p2


def main():
    df=pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    actual=[draw(r) for _,r in df.iterrows()]

    obs_whole=whole_recurrence_lift(actual)
    obs_struct=structure_recurrence_lift(actual)

    rng=random.Random(SEED)
    whole_vals=[]
    struct_vals=[]
    for _ in range(SIMS):
        sim=[frozenset(rng.sample(range(1,38),7)) for _ in range(len(actual))]
        whole_vals.append(whole_recurrence_lift(sim))
        struct_vals.append(structure_recurrence_lift(sim))

    rows=[]
    for name,obs,vals in [
        ("whole_transition_recurrence_lift",obs_whole,whole_vals),
        ("structure_recurrence_lift",obs_struct,struct_vals),
    ]:
        mu,sd,z,pct,p2=empirical_summary(vals,obs)
        rows.append({
            "metric":name,
            "observed":obs,
            "random_mean":mu,
            "random_sd":sd,
            "z":z,
            "percentile":pct,
            "two_sided_empirical_p":p2,
        })

    res=pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    res.to_csv(OUT,index=False)

    print("=== LOTO7 TRANSITION RECURRENCE RANDOM CONTROL v0 | PILOT200 ===")
    print("Null: each draw is an independent uniform 7-of-37 sample.")
    print(f"draws={len(actual)} sims={SIMS} seed={SEED}")
    print("No fitted thresholds or weights.")
    print()
    for _,r in res.iterrows():
        print(
            f"{r['metric']}: observed={r['observed']:+.6f} "
            f"random_mean={r['random_mean']:+.6f} sd={r['random_sd']:.6f} "
            f"z={r['z']:+.3f} percentile={r['percentile']:.3%} "
            f"p2={r['two_sided_empirical_p']:.3%}"
        )
    print()
    print(f"saved -> {OUT}")


if __name__=="__main__":
    main()
