from __future__ import annotations

from pathlib import Path
import pandas as pd

import lotocore
import lotocore_finite_generation_v0 as fg

DATA=Path("data/loto7.csv")
OUT=Path("results/loto7_core_finite_generation_v0_benchmark.csv")
WINDOW=100


def actual(row):
    return {int(row[f"n{i}"]) for i in range(1,8)}


def nearest_distance(truth, nums):
    vals=list(nums)
    return sum(min(abs(a-n) for n in vals) for a in truth)/7


def main():
    df=pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    rows=[]

    for i in range(WINDOW,len(df)):
        history=df.iloc[i-WINDOW:i]
        truth=actual(df.iloc[i])
        base=tuple(sorted(lotocore.predict(history).numbers))
        cand=tuple(sorted(fg.predict(history).numbers))

        bh=len(set(base)&truth)
        ch=len(set(cand)&truth)
        rows.append({
            "target_round":int(df.iloc[i]["round"]),
            "base_hits":bh,
            "candidate_hits":ch,
            "hit_delta":ch-bh,
            "base_distance":nearest_distance(truth,base),
            "candidate_distance":nearest_distance(truth,cand),
            "distance_delta":nearest_distance(truth,cand)-nearest_distance(truth,base),
            "overlap":len(set(base)&set(cand)),
            "base":"-".join(f"{n:02d}" for n in base),
            "candidate":"-".join(f"{n:02d}" for n in cand),
            "actual":"-".join(f"{n:02d}" for n in sorted(truth)),
        })

    res=pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    res.to_csv(OUT,index=False)

    print("=== CORE FINITE GENERATION v0 BENCHMARK ===")
    print(f"targets={len(res)} rounds={int(res.target_round.min())}..{int(res.target_round.max())}")
    print("Candidate uses only pre-target history and current Core finite representation.")
    print()
    print(f"mean hits: base={res.base_hits.mean():.6f} candidate={res.candidate_hits.mean():.6f} delta={(res.candidate_hits-res.base_hits).mean():+.6f}")
    print(f"3+: base={(res.base_hits>=3).mean():.6f} candidate={(res.candidate_hits>=3).mean():.6f}")
    print(f"4+: base={(res.base_hits>=4).mean():.6f} candidate={(res.candidate_hits>=4).mean():.6f}")
    print(f"5+: base={(res.base_hits>=5).mean():.6f} candidate={(res.candidate_hits>=5).mean():.6f}")
    print(f"improve/worse/same={(res.hit_delta>0).sum()}/{(res.hit_delta<0).sum()}/{(res.hit_delta==0).sum()}")
    print()
    print(f"mean nearest distance: base={res.base_distance.mean():.6f} candidate={res.candidate_distance.mean():.6f} delta={(res.candidate_distance-res.base_distance).mean():+.6f}")
    print(f"distance improve/worse/same={(res.distance_delta<0).sum()}/{(res.distance_delta>0).sum()}/{(res.distance_delta.abs()<1e-12).sum()}")
    print(f"mean base/candidate overlap={res.overlap.mean():.6f}/7")
    print()
    print("RECENT 20")
    print(res.tail(20).to_string(index=False))
    print(f"saved -> {OUT}")


if __name__=="__main__":
    main()
