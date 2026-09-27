from __future__ import annotations

from pathlib import Path
import pandas as pd

from probe_loto7_a_world_shape_aware_box_v0 import (
    DATA, LONG_WINDOW, RECENT_WINDOW, ALPHA,
    draws_from_df, frequency_distribution, mix_distribution,
    distribution_state, rank_numbers, choose_shape_aware,
)

OUT=Path("results/loto7_a_world_boundary_closure_curve_v0.csv")
MAX_K=10


def main():
    df=pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    draws=draws_from_df(df)

    states=[]
    for i in range(LONG_WINDOW,len(draws)):
        lp=frequency_distribution(draws[i-LONG_WINDOW:i])
        rp=frequency_distribution(draws[i-RECENT_WINDOW:i])
        p=mix_distribution(lp,rp)
        ranked=rank_numbers(p)
        ac,asp=distribution_state(p)
        core,_,_,_,_,_,_,_=choose_shape_aware(ranked,p,ac,asp)
        core=set(core)
        noncore=[n for n in ranked if n not in core]
        states.append({
            "round":int(df.iloc[i]["round"]),
            "core":core,
            "noncore_ranked":noncore,
        })

    rows=[]
    for k in range(MAX_K+1):
        full=0
        total_covered=0
        total_outside=0
        total_entering=0
        entering_in_boundary=0
        changing=0
        for j in range(len(states)-1):
            cur=states[j]
            nxt=states[j+1]
            boundary=set(cur["noncore_ranked"][:k])
            support=set(cur["core"])|boundary
            next_core=set(nxt["core"])
            covered=len(support & next_core)
            outside=7-covered
            entering=next_core-set(cur["core"])
            if entering:
                changing+=1
                total_entering+=len(entering)
                entering_in_boundary+=len(entering & boundary)
            total_covered+=covered
            total_outside+=outside
            full += int(covered==7)

        n=len(states)-1
        rows.append({
            "boundary_k":k,
            "support_size":7+k,
            "transitions":n,
            "full_reentry_rate":full/n,
            "mean_next_core_covered":total_covered/n,
            "mean_next_core_outside":total_outside/n,
            "changing_transitions":changing,
            "total_entering":total_entering,
            "entering_captured_by_boundary":entering_in_boundary,
            "entering_capture_rate":entering_in_boundary/total_entering if total_entering else 0.0,
        })

    res=pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    res.to_csv(OUT,index=False)

    print("=== LOTO7 A-WORLD BOUNDARY CLOSURE CURVE v0 ===")
    print(f"transitions={len(states)-1} alpha={ALPHA:.2f} core=shape-aware7 max_boundary={MAX_K}")
    print("No lottery target outcome is used. This measures A-world re-entry closure only.")
    print()
    print(res.to_string(index=False))
    print()
    print("MARGINAL FULL-REENTRY GAIN")
    prev=None
    for _,r in res.iterrows():
        gain=0.0 if prev is None else r["full_reentry_rate"]-prev
        print(f"k={int(r['boundary_k'])}: full={r['full_reentry_rate']:.6f} gain={gain:+.6f} entering_capture={r['entering_capture_rate']:.6f}")
        prev=r["full_reentry_rate"]
    print()
    print(f"saved -> {OUT}")


if __name__=="__main__":
    main()
