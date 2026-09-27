from __future__ import annotations

from collections import Counter
from pathlib import Path
import pandas as pd

from probe_loto7_a_world_shape_aware_box_v0 import (
    DATA, LONG_WINDOW, RECENT_WINDOW, ALPHA,
    draws_from_df, frequency_distribution, mix_distribution,
    distribution_state, rank_numbers, choose_shape_aware,
)

OUT=Path("results/loto7_a_world_finite_reentry_bridge_v0.csv")
BOUNDARY_K=3


def delta_map(prev_p, cur_p):
    return {n: cur_p[n]-prev_p[n] for n in cur_p}


def main():
    df=pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    draws=draws_from_df(df)

    states=[]
    prev_p=None
    for i in range(LONG_WINDOW,len(draws)):
        lp=frequency_distribution(draws[i-LONG_WINDOW:i])
        rp=frequency_distribution(draws[i-RECENT_WINDOW:i])
        p=mix_distribution(lp,rp)
        ranked=rank_numbers(p)
        ac,asp=distribution_state(p)
        core,_,_,_,_,_,_,_=choose_shape_aware(ranked,p,ac,asp)

        core=set(core)
        boundary=[n for n in ranked if n not in core][:BOUNDARY_K]
        support=set(core)|set(boundary)

        d=delta_map(prev_p,p) if prev_p is not None else {n:0.0 for n in p}
        states.append({
            "round":int(df.iloc[i]["round"]),
            "core":tuple(sorted(core)),
            "boundary":tuple(boundary),
            "support":support,
            "p":p,
            "delta":d,
        })
        prev_p=p

    rows=[]
    for j in range(len(states)-1):
        cur=states[j]
        nxt=states[j+1]
        cur_core=set(cur["core"])
        nxt_core=set(nxt["core"])
        boundary=set(cur["boundary"])
        support=set(cur["support"])

        retained=len(cur_core & nxt_core)
        next_from_boundary=len(boundary & nxt_core)
        next_outside=len(nxt_core-support)
        support_capture=len(support & nxt_core)

        entering=nxt_core-cur_core
        entering_from_boundary=entering & boundary
        entering_outside=entering-support

        pos_boundary={n for n in boundary if cur["delta"][n] > 0}
        flat_boundary={n for n in boundary if abs(cur["delta"][n]) <= 1e-12}
        neg_boundary={n for n in boundary if cur["delta"][n] < 0}

        rows.append({
            "from_round":cur["round"],
            "to_round":nxt["round"],
            "core":"-".join(f"{n:02d}" for n in sorted(cur_core)),
            "boundary":"-".join(f"{n:02d}" for n in cur["boundary"]),
            "next_core":"-".join(f"{n:02d}" for n in sorted(nxt_core)),
            "retained_core_count":retained,
            "next_core_from_boundary_count":next_from_boundary,
            "next_core_outside_support_count":next_outside,
            "next_core_covered_by_support_count":support_capture,
            "full_reentry":support_capture==7,
            "core_only_full_reentry":retained==7,
            "entering_count":len(entering),
            "entering_from_boundary_count":len(entering_from_boundary),
            "entering_outside_count":len(entering_outside),
            "boundary_positive_delta_count":len(pos_boundary),
            "boundary_flat_delta_count":len(flat_boundary),
            "boundary_negative_delta_count":len(neg_boundary),
            "entering_from_positive_boundary_count":len(entering_from_boundary & pos_boundary),
            "entering_from_flat_boundary_count":len(entering_from_boundary & flat_boundary),
            "entering_from_negative_boundary_count":len(entering_from_boundary & neg_boundary),
        })

    res=pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    res.to_csv(OUT,index=False)

    changing=res[res["entering_count"]>0].copy()
    boundary_enters=changing["entering_from_boundary_count"].sum()
    total_enters=changing["entering_count"].sum()

    print("=== LOTO7 A-WORLD FINITE REENTRY BRIDGE v0 ===")
    print(f"transitions={len(res)} alpha={ALPHA:.2f} core=shape-aware7 boundary={BOUNDARY_K}")
    print("No lottery target outcome is used. This tests re-entry inside A-world only.")
    print()
    print("REENTRY COVERAGE")
    print(f"core-only full reentry={res['core_only_full_reentry'].mean():.6f}")
    print(f"core+boundary full reentry={res['full_reentry'].mean():.6f}")
    print(f"mean next-core covered by current support={res['next_core_covered_by_support_count'].mean():.6f} / 7")
    print(f"mean next-core outside current support={res['next_core_outside_support_count'].mean():.6f} / 7")
    print()
    print("WHEN CORE CHANGES")
    print(f"changing transitions={len(changing)}/{len(res)} = {len(changing)/len(res):.6f}")
    print(f"total entering core numbers={int(total_enters)}")
    print(f"entering numbers already in boundary={int(boundary_enters)} = {boundary_enters/total_enters if total_enters else 0:.6f}")
    print(f"entering numbers outside support={int(changing['entering_outside_count'].sum())} = {changing['entering_outside_count'].sum()/total_enters if total_enters else 0:.6f}")
    print()
    print("TRANSITION VECTOR ON BOUNDARY")
    pos=int(changing['entering_from_positive_boundary_count'].sum())
    flat=int(changing['entering_from_flat_boundary_count'].sum())
    neg=int(changing['entering_from_negative_boundary_count'].sum())
    denom=pos+flat+neg
    print(f"boundary entrants with positive delta={pos}/{denom} = {pos/denom if denom else 0:.6f}")
    print(f"boundary entrants with flat delta={flat}/{denom} = {flat/denom if denom else 0:.6f}")
    print(f"boundary entrants with negative delta={neg}/{denom} = {neg/denom if denom else 0:.6f}")
    print()
    print("OUTSIDE-SUPPORT DISTRIBUTION")
    print(str(res["next_core_outside_support_count"].value_counts().sort_index().to_dict()))
    print()
    print("RECENT 20")
    cols=[
        "from_round","to_round","core","boundary","next_core",
        "retained_core_count","next_core_from_boundary_count",
        "next_core_outside_support_count","full_reentry",
        "entering_from_positive_boundary_count"
    ]
    print(res[cols].tail(20).to_string(index=False))
    print(f"saved -> {OUT}")


if __name__=="__main__":
    main()
