from __future__ import annotations

import csv, io, json
from collections import Counter
from pathlib import Path
import numpy as np
import requests

URL = "https://www.mk-mode.com/rails/loto/LOTO6_ALL.csv"
OUT = Path("results/loto6_route_temporal_stability_v0.json")

SEED = 20260925
BURN = 100
CENTER = 37 / 6
CENTER_BAND = 0.25
NULL_WORLDS = 180
CAL_DRAWS = 200000

PHASES = ["Flow","Break","Uncertainty","Re-formation"]
MOTIONS = ["Hold","Expand","Contract","Shift-L","Shift-R","Shift+Expand","Shift+Contract"]

WATCH_EDGE = [
    ("Break","Uncertainty","Expand"),
    ("Uncertainty","Uncertainty","Shift-R"),
    ("Flow","Break","Shift+Expand"),
    ("Break","Re-formation","Expand"),
    ("Break","Re-formation","Shift-L"),
    ("Re-formation","Flow","Shift+Contract"),
]
WATCH_TRIPLE = [
    ("Uncertainty","Break","Uncertainty","Contract","Expand"),
    ("Break","Break","Uncertainty","Shift-L","Shift-R"),
    ("Uncertainty","Break","Re-formation","Shift-R","Expand"),
    ("Break","Re-formation","Flow","Expand","Shift+Contract"),
]

def fetch_history():
    r=requests.get(URL,timeout=30,headers={"User-Agent":"lotocore-X route-temporal-stability"})
    r.raise_for_status()
    try: text=r.content.decode("cp932")
    except UnicodeDecodeError: text=r.content.decode("utf-8")
    rows=[]
    rd=csv.reader(io.StringIO(text)); next(rd,None)
    for row in rd:
        try:
            rnd=int(row[0]); nums=[int(row[2+i]) for i in range(6)]
        except Exception:
            continue
        if len(nums)==6 and len(set(nums))==6 and all(1<=n<=43 for n in nums):
            rows.append((rnd,sorted(nums)))
    rows.sort(key=lambda x:x[0])
    if len(rows)<1000: raise RuntimeError(len(rows))
    return rows

def random_world(rng,n):
    u=rng.random((n,43))
    idx=np.argpartition(u,6,axis=1)[:,:6]+1
    return np.sort(idx,axis=1)

def draw_shape(draw):
    a=np.asarray(draw,dtype=float)
    return float(a.mean()), float(a.var())

def calibrate(rng):
    x=random_world(rng,CAL_DRAWS)
    mus=x.mean(axis=1); vv=x.var(axis=1)
    return {
        "mu_mean":float(mus.mean()), "mu_sd":float(mus.std(ddof=1)),
        "var_mean":float(vv.mean()), "var_sd":float(vv.std(ddof=1)),
    }

def classify_phase(A,prev_A,prev_label):
    x=A-CENTER
    if abs(x)<=CENTER_BAND: return "Uncertainty"
    dA=0.0 if prev_A is None else A-prev_A
    centerward=(x*dA)<0
    if centerward and prev_label in {"Break","Uncertainty"}: return "Re-formation"
    if centerward: return "Flow"
    return "Break"

def motion_family(dmu,dvar):
    am,av=abs(dmu),abs(dvar); t=.35
    if am<=t and av<=t: return "Hold"
    if av>am*1.25: return "Expand" if dvar>0 else "Contract"
    if am>av*1.25: return "Shift-R" if dmu>0 else "Shift-L"
    if dvar>0: return "Shift+Expand"
    return "Shift+Contract"

def sequence(draws,cal):
    draws=np.asarray(draws,dtype=int)
    shapes=[draw_shape(d) for d in draws]
    ages=np.zeros(43,dtype=float)
    prev_A=None; prev_phase="Uncertainty"
    phases=[]; z=[]
    for i,draw in enumerate(draws):
        A=float(ages.mean())
        p=classify_phase(A,prev_A,prev_phase)
        if i==0:
            mu,vv=cal["mu_mean"],cal["var_mean"]
        else:
            mu,vv=shapes[i-1]
        zz=np.array([
            (mu-cal["mu_mean"])/max(cal["mu_sd"],1e-9),
            (vv-cal["var_mean"])/max(cal["var_sd"],1e-9)
        ])
        if i>=BURN:
            phases.append(p); z.append(zz)
        nxt=ages+1
        for n in draw: nxt[int(n)-1]=0
        ages=nxt; prev_A=A; prev_phase=p
    motions=[]
    for a,b in zip(z,z[1:]):
        d=b-a
        motions.append(motion_family(float(d[0]),float(d[1])))
    return phases,motions

def split_bounds(n):
    # 3 equal chronological blocks on post-burn phase sequence
    q1=n//3; q2=2*n//3
    return [(0,q1),(q1,q2),(q2,n)]

def block_metrics(phases,motions):
    bounds=split_bounds(len(phases))
    out=[]
    for bi,(lo,hi) in enumerate(bounds):
        # edge i uses phases[i],phases[i+1], motions[i]; require both phase endpoints inside block
        edge_den=Counter(); edge_num=Counter()
        for i in range(lo, min(hi-1,len(motions))):
            a,b=phases[i],phases[i+1]; m=motions[i]
            edge_den[(a,b)]+=1; edge_num[(a,b,m)]+=1
        tri_den=Counter(); tri_num=Counter()
        for i in range(lo, min(hi-2,len(motions)-1)):
            a,b,c=phases[i],phases[i+1],phases[i+2]
            m1,m2=motions[i],motions[i+1]
            tri_den[(a,b,c)]+=1; tri_num[(a,b,c,m1,m2)]+=1
        out.append({
            "block":bi+1,
            "phase_index_range":[lo,hi],
            "edge_den":edge_den,"edge_num":edge_num,
            "tri_den":tri_den,"tri_num":tri_num
        })
    return out

def rate_edge(block,key):
    a,b,m=key; den=block["edge_den"][(a,b)]
    return (block["edge_num"][(a,b,m)]/den if den else None, den)

def rate_tri(block,key):
    a,b,c,m1,m2=key; den=block["tri_den"][(a,b,c)]
    return (block["tri_num"][(a,b,c,m1,m2)]/den if den else None, den)

def summarize_watch(actual_blocks, world_blocks, keys, kind):
    rows=[]
    for key in keys:
        blocks=[]
        for bi in range(3):
            ar,aden=(rate_edge(actual_blocks[bi],key) if kind=="edge" else rate_tri(actual_blocks[bi],key))
            vals=[]; dens=[]
            for wb in world_blocks:
                rr,dd=(rate_edge(wb[bi],key) if kind=="edge" else rate_tri(wb[bi],key))
                if rr is not None:
                    vals.append(rr); dens.append(dd)
            if ar is None or not vals:
                blocks.append({"block":bi+1,"A_rate":ar,"A_opportunities":aden,"B_worlds":len(vals)})
                continue
            arr=np.asarray(vals,dtype=float)
            mean=float(arr.mean())
            blocks.append({
                "block":bi+1,
                "A_rate":float(ar),
                "A_opportunities":int(aden),
                "B_mean_rate":mean,
                "B_q10":float(np.quantile(arr,.10)),
                "B_q90":float(np.quantile(arr,.90)),
                "B_q025":float(np.quantile(arr,.025)),
                "B_q975":float(np.quantile(arr,.975)),
                "delta":float(ar-mean),
                "percentile":float(np.mean(arr<=ar)),
                "B_mean_opportunities":float(np.mean(dens)),
            })
        signs=[np.sign(x.get("delta",0.0)) for x in blocks if "delta" in x]
        same_sign=(len(signs)==3 and (all(s>0 for s in signs) or all(s<0 for s in signs)))
        rows.append({
            "key":" -> ".join(key[:2])+" | "+key[2] if kind=="edge"
                  else " -> ".join(key[:3])+" | "+key[3]+" -> "+key[4],
            "kind":kind,
            "blocks":blocks,
            "same_delta_sign_all_3":bool(same_sign),
            "edge_or_triple":list(key),
        })
    return rows

def global_scan(actual_blocks,world_blocks):
    edge_rows=[]
    for a in PHASES:
        for b in PHASES:
            for m in MOTIONS:
                key=(a,b,m); bs=[]
                valid=True
                for bi in range(3):
                    ar,aden=rate_edge(actual_blocks[bi],key)
                    vals=[]
                    for wb in world_blocks:
                        rr,_=rate_edge(wb[bi],key)
                        if rr is not None: vals.append(rr)
                    if ar is None or aden<20 or len(vals)<100:
                        valid=False; break
                    arr=np.asarray(vals,dtype=float); mean=float(arr.mean())
                    bs.append((float(ar-mean),float(np.mean(arr<=ar)),aden))
                if valid:
                    signs=[np.sign(x[0]) for x in bs]
                    if all(s>0 for s in signs) or all(s<0 for s in signs):
                        score=min(abs(x[0]) for x in bs)
                        edge_rows.append({
                            "key":[a,b,m],"blocks":[{"delta":x[0],"percentile":x[1],"A_opportunities":x[2]} for x in bs],
                            "min_abs_delta":score
                        })
    edge_rows.sort(key=lambda r:r["min_abs_delta"],reverse=True)

    tri_rows=[]
    for a in PHASES:
        for b in PHASES:
            for c in PHASES:
                for m1 in MOTIONS:
                    for m2 in MOTIONS:
                        key=(a,b,c,m1,m2); bs=[]; valid=True
                        for bi in range(3):
                            ar,aden=rate_tri(actual_blocks[bi],key)
                            vals=[]
                            for wb in world_blocks:
                                rr,_=rate_tri(wb[bi],key)
                                if rr is not None: vals.append(rr)
                            if ar is None or aden<12 or len(vals)<100:
                                valid=False; break
                            arr=np.asarray(vals,dtype=float); mean=float(arr.mean())
                            bs.append((float(ar-mean),float(np.mean(arr<=ar)),aden))
                        if valid:
                            signs=[np.sign(x[0]) for x in bs]
                            if all(s>0 for s in signs) or all(s<0 for s in signs):
                                score=min(abs(x[0]) for x in bs)
                                tri_rows.append({
                                    "key":[a,b,c,m1,m2],
                                    "blocks":[{"delta":x[0],"percentile":x[1],"A_opportunities":x[2]} for x in bs],
                                    "min_abs_delta":score
                                })
    tri_rows.sort(key=lambda r:r["min_abs_delta"],reverse=True)
    return {"edge_same_sign_top":edge_rows[:20],"triple_same_sign_top":tri_rows[:20]}

def main():
    rows=fetch_history()
    draws=np.asarray([x for _,x in rows],dtype=int)
    rng=np.random.default_rng(SEED)
    cal=calibrate(rng)

    p,m=sequence(draws,cal)
    actual_blocks=block_metrics(p,m)

    world_blocks=[]
    for _ in range(NULL_WORLDS):
        wp,wm=sequence(random_world(rng,len(rows)),cal)
        world_blocks.append(block_metrics(wp,wm))

    out={
        "probe":"LOTO6 Route Grammar Temporal Stability v0",
        "source":URL,
        "rounds":[rows[0][0],rows[-1][0]],
        "draws":len(rows),
        "post_burn_phase_points":len(p),
        "split":"3 equal chronological blocks after burn; no retuning per block",
        "question":"Do the conditional turning differences keep the same direction across time blocks?",
        "watched_from_previous_probe":{
            "edge":summarize_watch(actual_blocks,world_blocks,WATCH_EDGE,"edge"),
            "triple":summarize_watch(actual_blocks,world_blocks,WATCH_TRIPLE,"triple"),
        },
        "global_descriptive_scan":global_scan(actual_blocks,world_blocks),
        "null":{
            "worlds":NULL_WORLDS,"seed":SEED,
            "generator":"independent exact-uniform 6-of-43 worlds, split identically into thirds"
        },
        "boundary":[
            "Temporal consistency observation only; no predictor is fit.",
            "Watched routes were frozen from the prior normalization probe before this run.",
            "Global scan is descriptive and subject to multiple-comparison effects.",
            "Historical A is exploratory, not pristine holdout."
        ]
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
