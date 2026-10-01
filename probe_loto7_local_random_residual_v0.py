from __future__ import annotations
import argparse, random, itertools
from collections import defaultdict
from pathlib import Path
import pandas as pd
import lotocore

DATA=Path("data/loto7.csv")

def actual(row):
    return set(int(row[f"n{i}"]) for i in range(1,8))

def core18(history):
    snap=lotocore.score_snapshot(history)
    ranks={int(k):int(v) for k,v in snap["ranks"].items()}
    return [n for n,_ in sorted(ranks.items(), key=lambda kv:(kv[1],kv[0]))[:18]]

def structured(candidates):
    target={n:3 for n in candidates}
    for n in candidates[:16]: target[n]=4
    rem=target.copy(); pair=defaultdict(int); out=[]
    for slot in range(10):
        left=10-slot
        chosen=[n for n in candidates if rem[n]==left]
        while len(chosen)<7:
            pool=[n for n in candidates if rem[n]>0 and n not in chosen]
            def key(n):
                return (sum(pair[tuple(sorted((n,x)))] for x in chosen), -rem[n], candidates.index(n))
            chosen.append(min(pool,key=key))
        chosen=sorted(chosen); out.append(tuple(chosen))
        for n in chosen: rem[n]-=1
        for a,b in itertools.combinations(chosen,2): pair[tuple(sorted((a,b)))] += 1
    return out

def random_bundle(candidates,seed):
    rng=random.Random(seed); out=[]; seen=set()
    while len(out)<10:
        t=tuple(sorted(rng.sample(candidates,7)))
        if t not in seen: seen.add(t); out.append(t)
    return out

def local_random(bundle,candidates,seed,swaps):
    rng=random.Random(seed); out=[]
    for t in bundle:
        s=set(t)
        for _ in range(swaps):
            drop=rng.choice(sorted(s))
            add=rng.choice([n for n in candidates if n not in s])
            s.remove(drop); s.add(add)
        out.append(tuple(sorted(s)))
    return out

def maxhit(bundle,a): return max(len(set(t)&a) for t in bundle)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--window",type=int,default=100)
    ap.add_argument("--reps",type=int,default=100)
    args=ap.parse_args()
    df=pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    acc={k:[] for k in ["structured","random","local1","local2"]}
    for i in range(args.window,len(df)):
        h=df.iloc[i-args.window:i]; row=df.iloc[i]
        cand=core18(h); a=actual(row); base=structured(cand); rnd=int(row["round"])
        acc["structured"].append(maxhit(base,a))
        rr=[]; l1=[]; l2=[]
        for rep in range(args.reps):
            seed=rnd*1000003+rep*7919+17
            rr.append(maxhit(random_bundle(cand,seed),a))
            l1.append(maxhit(local_random(base,cand,seed+1,1),a))
            l2.append(maxhit(local_random(base,cand,seed+2,2),a))
        acc["random"].append(sum(rr)/len(rr))
        acc["local1"].append(sum(l1)/len(l1))
        acc["local2"].append(sum(l2)/len(l2))
    print("=== LOTO7 LOCAL RANDOM RESIDUAL v0 ===")
    print(f"targets={len(acc['structured'])} reps={args.reps}")
    for k,v in acc.items():
        if k=="structured":
            print(f"{k}: mean={sum(v)/len(v):.6f} 5+={sum(x>=5 for x in v)/len(v):.6f} 6+={sum(x>=6 for x in v)/len(v):.6f} 7={sum(x>=7 for x in v)/len(v):.6f}")
        else:
            # v already per-round probability/expected max only for max-hit average, so recompute rates separately is needed.
            pass
    # second pass for exact reported rates
    rows=[]
    for i in range(args.window,len(df)):
        h=df.iloc[i-args.window:i]; row=df.iloc[i]
        cand=core18(h); a=actual(row); base=structured(cand); rnd=int(row["round"])
        rec={"round":rnd}
        for name, maker in [
            ("random", lambda seed: random_bundle(cand,seed)),
            ("local1", lambda seed: local_random(base,cand,seed+1,1)),
            ("local2", lambda seed: local_random(base,cand,seed+2,2)),
        ]:
            hs=[maxhit(maker(rnd*1000003+rep*7919+17),a) for rep in range(args.reps)]
            rec[name+"_mean"]=sum(hs)/len(hs)
            for z in (5,6,7): rec[f"{name}_{z}"]=sum(x>=z for x in hs)/len(hs)
        rec["structured_mean"]=maxhit(base,a)
        for z in (5,6,7): rec[f"structured_{z}"]=int(maxhit(base,a)>=z)
        rows.append(rec)
    out=pd.DataFrame(rows)
    for name in ["structured","random","local1","local2"]:
        print(f"{name}: mean={out[name+'_mean'].mean():.6f} 5+={out[name+'_5'].mean():.6f} 6+={out[name+'_6'].mean():.6f} 7={out[name+'_7'].mean():.6f}")
    Path("results").mkdir(exist_ok=True)
    out.to_csv("results/loto7_local_random_residual_v0.csv",index=False)

if __name__=="__main__": main()
