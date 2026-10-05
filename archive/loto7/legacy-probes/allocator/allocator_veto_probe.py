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

def pre_features(hist):
    xs=x_agent.score_snapshot(hist,competition_gate=True)
    cs=lotocore.score_snapshot(hist)
    xr,xscore=ranked_snapshot(xs)
    cr,_=ranked_snapshot(cs)
    vals=[max(xscore[n],0.0) for n in xr]
    total=sum(vals)
    st=xs["state"]
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
    },xr[:10]

def build_rows(df,start=201,end=550,seeds=SEEDS):
    by_round={int(r["round"]):i for i,r in df.iterrows()}
    rows=[]
    for rnd in range(start,end+1):
        if rnd not in by_round:continue
        idx=by_round[rnd]
        if idx<100:continue
        hist=df.iloc[idx-100:idx]
        feats,pool=pre_features(hist)
        main=main_numbers(df.iloc[idx])
        capture=len(set(pool)&main)
        bmax=[];nmax=[];wins=ties=losses=0
        for seed in range(1,seeds+1):
            b=bundle_eval(baseline_bundle(pool,seed),main)
            n=bundle_eval(narrow_bundle(pool,seed),main)
            bmax.append(b[0]);nmax.append(n[0])
            if n[0]>b[0]:wins+=1
            elif n[0]<b[0]:losses+=1
            else:ties+=1
        rows.append({
            "round":rnd,"capture":capture,**feats,
            "baseline_mean_max":sum(bmax)/seeds,
            "narrow_mean_max":sum(nmax)/seeds,
            "delta_mean_max":sum(nmax)/seeds-sum(bmax)/seeds,
            "paired_wins":wins,"paired_ties":ties,"paired_losses":losses,
            "paired_loss_rate":losses/seeds,
        })
    out=pd.DataFrame(rows).sort_values("round").reset_index(drop=True)
    return attach_instability(out)

def attach_instability(df):
    x=df.copy()
    # Every rolling outcome feature is shifted one round: only already-known draws enter.
    prior_delta=x["delta_mean_max"].shift(1)
    prior_loss=x["paired_loss_rate"].shift(1)
    prior_capture=x["capture"].shift(1)

    x["prior_delta_mean20"]=prior_delta.rolling(20,min_periods=10).mean()
    x["prior_delta_var20"]=prior_delta.rolling(20,min_periods=10).var(ddof=0)
    x["prior_loss_mean20"]=prior_loss.rolling(20,min_periods=10).mean()
    x["prior_loss_var20"]=prior_loss.rolling(20,min_periods=10).var(ddof=0)
    x["prior_capture_mean20"]=prior_capture.rolling(20,min_periods=10).mean()
    x["prior_capture_var20"]=prior_capture.rolling(20,min_periods=10).var(ddof=0)
    x["capture_shift_5_vs_20"]=(prior_capture.rolling(5,min_periods=5).mean()-prior_capture.rolling(20,min_periods=10).mean()).abs()

    for f in ["x_top10_mass","x_margin_10_11","x_entropy","competition_margin",
              "spacing_strength","action_signal","field_center_abs",
              "field_spread_abs","gap_abs","core_x_overlap10"]:
        prev=x[f].shift(1)
        x[f+"_shift_from5"]=(x[f]-prev.rolling(5,min_periods=3).mean()).abs()
        x[f+"_prior_var10"]=prev.rolling(10,min_periods=5).var(ddof=0)
    return x

FEATURES=[
    "prior_delta_var20","prior_loss_mean20","prior_loss_var20",
    "prior_capture_var20","capture_shift_5_vs_20",
    "x_top10_mass_shift_from5","x_margin_10_11_shift_from5","x_entropy_shift_from5",
    "competition_margin_shift_from5","spacing_strength_shift_from5","action_signal_shift_from5",
    "field_center_abs_shift_from5","field_spread_abs_shift_from5","gap_abs_shift_from5",
    "core_x_overlap10_shift_from5",
    "x_top10_mass_prior_var10","x_margin_10_11_prior_var10","x_entropy_prior_var10",
    "competition_margin_prior_var10","spacing_strength_prior_var10","action_signal_prior_var10",
    "field_center_abs_prior_var10","field_spread_abs_prior_var10","gap_abs_prior_var10",
    "core_x_overlap10_prior_var10",
]

def veto_metrics(frame,feature,op,thr):
    vals=frame[feature]
    veto=(vals>=thr) if op==">=" else (vals<=thr)
    valid=vals.notna()
    f=frame[valid].copy()
    veto=veto[valid]

    harmed=f["delta_mean_max"]<0
    improved=f["delta_mean_max"]>0
    vetoed=f[veto]
    allowed=f[~veto]

    harm_total=max(int(harmed.sum()),1)
    imp_total=max(int(improved.sum()),1)
    veto_n=max(int(veto.sum()),1)
    allowed_n=max(int((~veto).sum()),1)

    harm_captured=int((veto & harmed).sum())
    opportunity_blocked=int((veto & improved).sum())
    residual_harm=int(((~veto)&harmed).sum())

    mean_saved=float((-vetoed.loc[vetoed["delta_mean_max"]<0,"delta_mean_max"]).mean()) if (vetoed["delta_mean_max"]<0).any() else 0.0
    mean_forfeit=float(vetoed.loc[vetoed["delta_mean_max"]>0,"delta_mean_max"].mean()) if (vetoed["delta_mean_max"]>0).any() else 0.0

    return {
        "n":len(f),
        "vetoed":int(veto.sum()),
        "veto_rate":float(veto.mean()),
        "harm_total":int(harmed.sum()),
        "improved_total":int(improved.sum()),
        "harm_captured":harm_captured,
        "harm_capture_rate":harm_captured/harm_total,
        "opportunity_blocked":opportunity_blocked,
        "opportunity_block_rate":opportunity_blocked/imp_total,
        "veto_precision":harm_captured/veto_n,
        "residual_harm":residual_harm,
        "residual_harm_rate":residual_harm/allowed_n,
        "overall_harm_rate":int(harmed.sum())/max(len(f),1),
        "mean_harm_saved":mean_saved,
        "mean_gain_forfeited":mean_forfeit,
    }

def candidate_rules(design):
    rows=[]
    for f in FEATURES:
        s=design[f].dropna()
        if len(s)<60:continue
        for q in (.25,.33,.5,.67,.75):
            thr=float(s.quantile(q))
            for op in (">=","<="):
                m=veto_metrics(design,f,op,thr)
                if m["vetoed"]<15:continue
                # Veto is allowed to be conservative, but not to ban most rounds.
                if m["veto_rate"]>0.40:continue
                # Design ranking: capture harm, penalize lost opportunity and broad vetoing.
                score=(m["harm_capture_rate"]
                       -0.9*m["opportunity_block_rate"]
                       -0.4*m["veto_rate"]
                       -0.3*m["residual_harm_rate"])
                rows.append((score,f,op,thr,m))
    return sorted(rows,key=lambda z:(z[0],z[4]["harm_capture_rate"],-z[4]["opportunity_block_rate"]),reverse=True)

def passes_gate(m):
    # Fixed before validation is read.
    if m["veto_rate"]>0.35:return False
    if m["harm_capture_rate"]<0.40:return False
    if m["opportunity_block_rate"]>0.20:return False
    if m["residual_harm_rate"]>=0.75*m["overall_harm_rate"]:return False
    return True

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,default=Path("data/loto7.csv"))
    p.add_argument("--out",type=Path,default=Path("results/loto7_allocator_veto_probe.csv"))
    p.add_argument("--summary",type=Path,default=Path("results/loto7_allocator_veto_summary.txt"))
    p.add_argument("--seeds",type=int,default=100)
    a=p.parse_args()

    df=pd.read_csv(a.data).sort_values("round").reset_index(drop=True)
    rows=build_rows(df,201,550,a.seeds)
    design=rows[rows["round"].between(201,400)].copy()
    valid=rows[rows["round"].between(401,550)].copy()

    cands=candidate_rules(design)
    if not cands:
        raise RuntimeError("no veto candidates")
    score,f,op,thr,dm=cands[0]
    vm=veto_metrics(valid,f,op,thr)

    rows["split"]="validation"
    rows.loc[rows["round"].between(201,400),"split"]="design"
    rows["veto_rule"]=0
    mask=rows[f].notna()
    if op==">=":
        rows.loc[mask,"veto_rule"]=(rows.loc[mask,f]>=thr).astype(int)
    else:
        rows.loc[mask,"veto_rule"]=(rows.loc[mask,f]<=thr).astype(int)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    rows.to_csv(a.out,index=False)

    lines=[
        "=== LOTO7 ALLOCATOR VETO PROBE v0 ===",
        f"seeds={a.seeds}",
        "splits: design=201..400 validation=401..550; future shadow holdout starts at round 696",
        "role: VETO only blocks Narrow. Non-veto does NOT mean Narrow is safe or selected.",
        f"selected_veto_rule: {f} {op} {thr:.8f}",
        f"design: veto={dm['vetoed']}/{dm['n']} ({dm['veto_rate']:.3f}) harm_capture={dm['harm_captured']}/{dm['harm_total']} ({dm['harm_capture_rate']:.3f}) opportunity_block={dm['opportunity_blocked']}/{dm['improved_total']} ({dm['opportunity_block_rate']:.3f}) residual_harm_rate={dm['residual_harm_rate']:.3f} overall_harm_rate={dm['overall_harm_rate']:.3f} pass={passes_gate(dm)}",
        f"validation: veto={vm['vetoed']}/{vm['n']} ({vm['veto_rate']:.3f}) harm_capture={vm['harm_captured']}/{vm['harm_total']} ({vm['harm_capture_rate']:.3f}) opportunity_block={vm['opportunity_blocked']}/{vm['improved_total']} ({vm['opportunity_block_rate']:.3f}) residual_harm_rate={vm['residual_harm_rate']:.3f} overall_harm_rate={vm['overall_harm_rate']:.3f} pass={passes_gate(vm)}",
        f"validation mean_harm_saved={vm['mean_harm_saved']:.4f} mean_gain_forfeited={vm['mean_gain_forfeited']:.4f}",
        "PROMOTION_GATE: veto_rate<=0.35, harm_capture>=0.40, opportunity_block<=0.20, residual_harm_rate < 75% of overall_harm_rate in BOTH design and validation.",
        f"PROMOTE={passes_gate(dm) and passes_gate(vm)}",
        "BOUNDARY: all outcome-derived rolling features are shifted; current-round target never enters current-round Veto features.",
        "BOUNDARY: this is a danger filter only. It does not select Narrow.",
    ]
    a.summary.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines))

if __name__=="__main__":
    main()
