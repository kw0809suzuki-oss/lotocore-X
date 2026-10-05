#!/usr/bin/env python3
from __future__ import annotations

import argparse, math, random, sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lotocore
import x_agent

DRAW_SIZE=7
TICKETS=10
SEEDS=100

def main_numbers(row):
    return {int(row[f"n{i}"]) for i in range(1,8)}

def entropy(vals):
    xs=[max(float(v),0.0) for v in vals]
    s=sum(xs)
    if s<=0:return 0.0
    ps=[x/s for x in xs if x>0]
    return -sum(p*math.log(p) for p in ps)

def ranked_snapshot(snap):
    scores={int(k):float(v) for k,v in snap["scores"].items()}
    ranks={int(k):int(v) for k,v in snap["ranks"].items()}
    ranked=sorted(range(1,38),key=lambda n:(ranks[n],n))
    return ranked,scores

def rank_degrees(pool_order):
    total=TICKETS*DRAW_SIZE
    deg=Counter({n:1 for n in pool_order})
    rank={n:i for i,n in enumerate(pool_order)}
    weight={n:len(pool_order)-i for i,n in enumerate(pool_order)}
    for _ in range(total-len(pool_order)):
        candidates=[n for n in pool_order if deg[n]<TICKETS]
        pick=max(candidates,key=lambda n:(weight[n]/(deg[n]+1.0),-rank[n]))
        deg[pick]+=1
    return deg

def baseline_bundle(pool_order,seed):
    remaining=Counter(rank_degrees(pool_order))
    pair_count=defaultdict(int)
    rank={n:i for i,n in enumerate(pool_order)}
    rng=random.Random(seed)
    out=[]
    for i in range(TICKETS):
        left=TICKETS-i
        chosen=[n for n,c in remaining.items() if c==left]
        while len(chosen)<DRAW_SIZE:
            candidates=[n for n,c in remaining.items() if c>0 and n not in chosen]
            rng.shuffle(candidates)
            pick=min(candidates,key=lambda n:(
                sum(pair_count[tuple(sorted((n,x)))] for x in chosen),
                -remaining[n],rank[n]))
            chosen.append(pick)
        for n in chosen:remaining[n]-=1
        t=tuple(sorted(chosen))
        for a in range(DRAW_SIZE):
            for b in range(a+1,DRAW_SIZE):
                pair_count[(t[a],t[b])]+=1
        out.append(t)
    return out

def narrow_bundle(pool_order,seed):
    rng=random.Random(seed)
    out=[];seen=set()
    while len(out)<TICKETS:
        t=tuple(sorted(rng.sample(pool_order,DRAW_SIZE)))
        if t not in seen:
            seen.add(t);out.append(t)
    return out

def bundle_eval(tickets,main):
    hits=[len(set(t)&main) for t in tickets]
    m=max(hits)
    return m,int(m>=5),int(m>=6)

def pre_features(hist,prior_capture):
    xs=x_agent.score_snapshot(hist,competition_gate=True)
    cs=lotocore.score_snapshot(hist)
    xr,xscore=ranked_snapshot(xs)
    cr,_=ranked_snapshot(cs)
    vals=[max(xscore[n],0.0) for n in xr]
    total=sum(vals)
    st=xs["state"]
    prev=prior_capture[-20:]
    return {
        "x_top10_mass":sum(vals[:10])/total if total else 0.0,
        "x_margin_10_11":xscore[xr[9]]-xscore[xr[10]],
        "x_entropy":entropy(vals),
        "competition_margin":float(st["competition_margin"]),
        "spacing_strength":float(st["spacing_strength"]),
        "action_signal":float(st["action_signal"]),
        "field_center_abs":abs(float(st["field_center_delta"])),
        "field_spread_abs":abs(float(st["field_spread_delta"])),
        "gap_abs":abs(float(st["gap_delta"])),
        "core_x_overlap10":len(set(xr[:10])&set(cr[:10])),
        "prior_x_capture_mean20":sum(prev)/len(prev) if prev else float("nan"),
        "prior_x_capture_ge3_rate20":sum(v>=3 for v in prev)/len(prev) if prev else float("nan"),
    },xr[:10]

def build_rows(df,start=201,end=695,seeds=SEEDS):
    by_round={int(r["round"]):i for i,r in df.iterrows()}
    prior_capture=[]
    rows=[]
    for rnd in range(start,end+1):
        if rnd not in by_round:continue
        idx=by_round[rnd]
        if idx<100:continue
        hist=df.iloc[idx-100:idx]
        feats,pool=pre_features(hist,prior_capture)
        main=main_numbers(df.iloc[idx])
        capture=len(set(pool)&main)
        bmax=[];nmax=[];b5=[];n5=[];b6=[];n6=[]
        wins=ties=losses=0
        for seed in range(1,seeds+1):
            b= bundle_eval(baseline_bundle(pool,seed),main)
            n= bundle_eval(narrow_bundle(pool,seed),main)
            bmax.append(b[0]);nmax.append(n[0]);b5.append(b[1]);n5.append(n[1]);b6.append(b[2]);n6.append(n[2])
            if n[0]>b[0]:wins+=1
            elif n[0]<b[0]:losses+=1
            else:ties+=1
        rec={"round":rnd,"capture":capture,**feats,
             "baseline_mean_max":sum(bmax)/seeds,"narrow_mean_max":sum(nmax)/seeds,
             "delta_mean_max":sum(nmax)/seeds-sum(bmax)/seeds,
             "baseline_reach5":sum(b5)/seeds,"narrow_reach5":sum(n5)/seeds,
             "baseline_reach6":sum(b6)/seeds,"narrow_reach6":sum(n6)/seeds,
             "paired_wins":wins,"paired_ties":ties,"paired_losses":losses,
             "loss_rate":losses/seeds}
        rows.append(rec)
        prior_capture.append(capture)
    return pd.DataFrame(rows)

FEATURES=[
"x_top10_mass","x_margin_10_11","x_entropy","competition_margin",
"spacing_strength","action_signal","field_center_abs","field_spread_abs",
"gap_abs","core_x_overlap10","prior_x_capture_mean20","prior_x_capture_ge3_rate20"]

def eval_policy(frame,feature,op,thr):
    vals=frame[feature]
    choose=(vals>=thr) if op==">=" else (vals<=thr)
    chosen=frame[choose]
    # Selector: Narrow if rule true, otherwise Baseline.
    selector_mean=(frame["baseline_mean_max"] + choose.astype(float)*frame["delta_mean_max"]).mean()
    base_mean=frame["baseline_mean_max"].mean()
    improved=((choose)&(frame["delta_mean_max"]>0)).sum()
    harmed=((choose)&(frame["delta_mean_max"]<0)).sum()
    return {
        "n":len(frame),"selected":int(choose.sum()),"support":float(choose.mean()),
        "base_mean":float(base_mean),"selector_mean":float(selector_mean),
        "delta":float(selector_mean-base_mean),
        "improved":int(improved),"harmed":int(harmed),
        "selected_loss_rate":float(chosen["loss_rate"].mean()) if len(chosen) else 0.0,
        "selected_narrow5":float(chosen["narrow_reach5"].mean()) if len(chosen) else 0.0,
        "selected_base5":float(chosen["baseline_reach5"].mean()) if len(chosen) else 0.0,
    }

def choose_rule(design):
    candidates=[]
    for f in FEATURES:
        s=design[f].dropna()
        if len(s)<40:continue
        for q in (.33,.5,.67):
            thr=float(s.quantile(q))
            for op in (">=","<="):
                m=eval_policy(design,f,op,thr)
                if m["selected"]<20:continue
                # Conservative: reward mean gain, penalize harm count and loss rate.
                score=m["delta"] - 0.01*(m["harmed"]/max(m["selected"],1)) - 0.05*m["selected_loss_rate"]
                candidates.append((score,f,op,thr,m))
    return max(candidates,key=lambda x:(x[0],x[4]["delta"],-x[4]["harmed"]))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,default=Path("data/loto7.csv"))
    p.add_argument("--out",type=Path,default=Path("results/loto7_allocator_selector_probe.csv"))
    p.add_argument("--summary",type=Path,default=Path("results/loto7_allocator_selector_summary.txt"))
    p.add_argument("--seeds",type=int,default=100)
    a=p.parse_args()
    df=pd.read_csv(a.data).sort_values("round").reset_index(drop=True)
    rows=build_rows(df,201,550,a.seeds)
    design=rows[(rows["round"]>=201)&(rows["round"]<=400)].copy()
    valid=rows[(rows["round"]>=401)&(rows["round"]<=550)].copy()
    holdout=rows[(rows["round"]>=551)&(rows["round"]<=695)].copy()

    score,f,op,thr,dm=choose_rule(design)
    vm=eval_policy(valid,f,op,thr)

    rows["split"]="validation"
    rows.loc[rows["round"].between(201,400),"split"]="design"
    rows.loc[rows["round"].between(401,550),"split"]="validation"
    rows["selector_rule_selected_in_design"]=0
    rows.loc[rows["split"].isin(["design","validation"]),"selector_rule_selected_in_design"]=(
        (rows.loc[rows["split"].isin(["design","validation"]),f]>=thr) if op==">="
        else (rows.loc[rows["split"].isin(["design","validation"]),f]<=thr)
    ).astype(int)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    rows.to_csv(a.out,index=False)

    lines=[
      "=== LOTO7 ALLOCATOR SELECTOR PROBE v0 ===",
      f"seeds={a.seeds}",
      "splits: design=201..400 validation=401..550. No historical final holdout is claimed in v0.",
      f"selected_rule: {f} {op} {thr:.8f}",
      f"design: selected={dm['selected']}/{dm['n']} delta_mean_max={dm['delta']:+.6f} improved={dm['improved']} harmed={dm['harmed']} selected_loss_rate={dm['selected_loss_rate']:.4f} 5+ narrow/base={dm['selected_narrow5']:.4f}/{dm['selected_base5']:.4f}",
      f"validation: selected={vm['selected']}/{vm['n']} delta_mean_max={vm['delta']:+.6f} improved={vm['improved']} harmed={vm['harmed']} selected_loss_rate={vm['selected_loss_rate']:.4f} 5+ narrow/base={vm['selected_narrow5']:.4f}/{vm['selected_base5']:.4f}",
      "BOUNDARY: target outcomes are used only to score allocator performance and choose the rule inside DESIGN. No target outcome enters selector features.",
      "BOUNDARY: an earlier debug run computed 551..695 outcomes, so that interval is not treated as untouched final holdout. Future rounds 696+ are the clean holdout.",
      "BOUNDARY: this is a one-feature threshold probe, not a promoted production selector.",
    ]
    a.summary.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines))

if __name__=="__main__":
    main()
