from __future__ import annotations

from pathlib import Path
import pandas as pd
import lotocore
import lotocore_finite_generation_v0 as fg0
import lotocore_finite_generation_minimal_v1 as fg1

DATA=Path("data/loto7.csv")
OUT=Path("results/loto7_core_fg_minimal_v1_benchmark.csv")
WINDOW=100

def truth(row):
    return {int(row[f"n{i}"]) for i in range(1,8)}

def dist(t,nums):
    vals=list(nums)
    return sum(min(abs(a-n) for n in vals) for a in t)/7

def main():
    df=pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    rows=[]
    for i in range(WINDOW,len(df)):
        h=df.iloc[i-WINDOW:i]; t=truth(df.iloc[i])
        base=tuple(sorted(lotocore.predict(h).numbers))
        v0=tuple(sorted(fg0.predict(h).numbers))
        v1p=fg1.predict(h); v1=tuple(sorted(v1p.numbers))
        rows.append({
          "round":int(df.iloc[i]["round"]),
          "base_hits":len(set(base)&t),
          "v0_hits":len(set(v0)&t),
          "v1_hits":len(set(v1)&t),
          "base_dist":dist(t,base),
          "v0_dist":dist(t,v0),
          "v1_dist":dist(t,v1),
          "v1_changed":int(v1!=base),
          "v1_overlap":len(set(base)&set(v1)),
        })
    r=pd.DataFrame(rows); OUT.parent.mkdir(parents=True,exist_ok=True); r.to_csv(OUT,index=False)
    print("=== CORE FG MINIMAL v1 BENCHMARK ===")
    print(f"targets={len(r)}")
    for name in ("base","v0","v1"):
        print(f"{name}: mean_hits={r[f'{name}_hits'].mean():.6f} 3+={(r[f'{name}_hits']>=3).mean():.6f} 4+={(r[f'{name}_hits']>=4).mean():.6f} 5+={(r[f'{name}_hits']>=5).mean():.6f} distance={r[f'{name}_dist'].mean():.6f}")
    print()
    print(f"v1 vs base hit improve/worse/same={(r.v1_hits>r.base_hits).sum()}/{(r.v1_hits<r.base_hits).sum()}/{(r.v1_hits==r.base_hits).sum()}")
    print(f"v1 vs base distance improve/worse/same={(r.v1_dist<r.base_dist).sum()}/{(r.v1_dist>r.base_dist).sum()}/{(r.v1_dist==r.base_dist).sum()}")
    print(f"v1 changed={r.v1_changed.sum()}/{len(r)} = {r.v1_changed.mean():.6f}")
    print(f"v1 mean overlap={r.v1_overlap.mean():.6f}/7")
    print(f"saved -> {OUT}")

if __name__=="__main__":
    main()
