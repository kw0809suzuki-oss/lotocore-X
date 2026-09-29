from __future__ import annotations

from pathlib import Path
import math
import statistics

import pandas as pd

DATA = Path("data/loto7.csv")
OUT = Path("results/loto7_transition_structure_ablation_v0.csv")


def draw(row):
    return sorted(int(row[f"n{i}"]) for i in range(1, 8))


def center(xs):
    return sum(xs) / len(xs)


def spread(xs):
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


def transition_structure(a,b):
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


GROUPS={
    "COUNT":["stay_count","enter_count","exit_count"],
    "SHAPE":["center_delta","spread_delta","range_delta","max_gap_delta","gap_cv_delta"],
    "BAND":["low_delta","mid_delta","high_delta"],
    "SHAPE+BAND":["center_delta","spread_delta","range_delta","max_gap_delta","gap_cv_delta","low_delta","mid_delta","high_delta"],
    "ALL":["stay_count","enter_count","exit_count","center_delta","spread_delta","range_delta","max_gap_delta","gap_cv_delta","low_delta","mid_delta","high_delta"],
}

SCALES={
    "stay_count":7.0,"enter_count":7.0,"exit_count":7.0,
    "center_delta":18.0,"spread_delta":10.0,"range_delta":36.0,
    "max_gap_delta":36.0,"gap_cv_delta":2.0,
    "low_delta":7.0,"mid_delta":7.0,"high_delta":7.0,
}


def dist(x,y,features):
    return sum(abs(x[f]-y[f])/SCALES[f] for f in features)/len(features)


def evaluate(structs,features):
    rows=[]
    for i in range(2,len(structs)-1):
        eligible=list(range(0,i-1))
        if not eligible:
            continue
        j=min(eligible,key=lambda k:(dist(structs[i],structs[k],features),k))
        next_d=dist(structs[i+1],structs[j+1],features)
        controls=[dist(structs[i+1],structs[k+1],features) for k in eligible]
        control=statistics.mean(controls)
        rows.append((control-next_d,dist(structs[i],structs[j],features),next_d,control))
    return rows


def main():
    df=pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    draws=[draw(r) for _,r in df.iterrows()]
    structs=[transition_structure(draws[i],draws[i+1]) for i in range(len(draws)-1)]

    out_rows=[]
    print("=== LOTO7 TRANSITION STRUCTURE ABLATION v0 ===")
    print("Purpose: isolate which structural family carries recurrence.")
    print("No numbers, no selector, no prediction rule, no fitted thresholds/weights.")
    print()

    for name,features in GROUPS.items():
        rows=evaluate(structs,features)
        lifts=[r[0] for r in rows]
        cur=[r[1] for r in rows]
        nxt=[r[2] for r in rows]
        ctrl=[r[3] for r in rows]
        n=len(rows)
        cut=int(n*0.60)
        early=lifts[:cut]
        late=lifts[cut:]

        rec={
            "group":name,
            "features":"|".join(features),
            "n":n,
            "mean_current_distance":statistics.mean(cur),
            "mean_next_distance":statistics.mean(nxt),
            "mean_control_distance":statistics.mean(ctrl),
            "mean_lift":statistics.mean(lifts),
            "median_lift":statistics.median(lifts),
            "positive":sum(x>0 for x in lifts),
            "same":sum(x==0 for x in lifts),
            "negative":sum(x<0 for x in lifts),
            "early60_mean_lift":statistics.mean(early),
            "late40_mean_lift":statistics.mean(late),
        }
        out_rows.append(rec)

        print(f"--- {name} ---")
        print(f"features={','.join(features)}")
        print(f"mean_lift={rec['mean_lift']:+.6f} median={rec['median_lift']:+.6f}")
        print(f"positive/same/negative={rec['positive']}/{rec['same']}/{rec['negative']}")
        print(f"early60={rec['early60_mean_lift']:+.6f} late40={rec['late40_mean_lift']:+.6f}")
        print()

    res=pd.DataFrame(out_rows)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    res.to_csv(OUT,index=False)

    print("ORDER BY MEAN LIFT")
    print(res.sort_values("mean_lift",ascending=False)[[
        "group","mean_lift","median_lift","positive","negative","early60_mean_lift","late40_mean_lift"
    ]].to_string(index=False))
    print()
    print(f"saved -> {OUT}")


if __name__=="__main__":
    main()
