from __future__ import annotations

import csv, io, json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import requests

URL = "https://www.mk-mode.com/rails/loto/LOTO6_ALL.csv"
OUT = Path("results/loto6_flow_roads_v0.json")

SEED = 20260924
BURN = 100
CENTER = 37/6
CENTER_BAND = 0.25
NULL_WORLDS = 120
CAL_DRAWS = 200000


def fetch_history():
    r = requests.get(URL, timeout=30, headers={"User-Agent":"lotocore-X flow-roads"})
    r.raise_for_status()
    try: text = r.content.decode("cp932")
    except UnicodeDecodeError: text = r.content.decode("utf-8")
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


def shp(draw):
    a=np.asarray(draw,dtype=float)
    return float(a.mean()), float(a.var())


def calibrate(rng):
    x=random_world(rng,CAL_DRAWS)
    mus=x.mean(axis=1); vv=x.var(axis=1)
    return {
        "mu_mean":float(mus.mean()),"mu_sd":float(mus.std(ddof=1)),
        "var_mean":float(vv.mean()),"var_sd":float(vv.std(ddof=1))
    }


def phase(A,prev_A,prev_label):
    x=A-CENTER
    if abs(x)<=CENTER_BAND: return "U"
    dA=0.0 if prev_A is None else A-prev_A
    centerward=(x*dA)<0
    if centerward and prev_label in {"B","U"}: return "R"
    if centerward: return "F"
    return "B"


def compress(seq):
    out=[]
    for x in seq:
        if not out or out[-1]!=x: out.append(x)
    return out


def evaluate(draws,cal,rounds=None):
    draws=np.asarray(draws,dtype=int)
    shapes=[shp(d) for d in draws]
    ages=np.zeros(43,dtype=float)
    prev_A=None; prev_label="U"
    states=[]

    for i,draw in enumerate(draws):
        A=float(ages.mean())
        p=phase(A,prev_A,prev_label)
        if i==0:
            mu,var=cal["mu_mean"],cal["var_mean"]
        else:
            mu,var=shapes[i-1]
        zmu=(mu-cal["mu_mean"])/max(cal["mu_sd"],1e-9)
        zv=(var-cal["var_mean"])/max(cal["var_sd"],1e-9)
        if i>=BURN:
            states.append({
                "round":int(rounds[i]) if rounds is not None else i+1,
                "phase":p,"zmu":float(zmu),"zv":float(zv)
            })
        nxt=ages+1
        for n in draw: nxt[int(n)-1]=0
        ages=nxt; prev_A=A; prev_label=p

    # Roads = from an F state, leave F, and stop at first later F state.
    roads=[]
    i=0
    while i < len(states)-2:
        if states[i]["phase"]!="F":
            i+=1; continue
        j=i+1
        # Skip F plateau; road starts at last F before leaving
        while j<len(states) and states[j]["phase"]=="F":
            j+=1
        if j>=len(states): break
        start=j-1
        k=j
        while k<len(states) and states[k]["phase"]!="F":
            k+=1
        if k>=len(states): break
        seg=states[start:k+1]
        ph=compress([s["phase"] for s in seg])
        template=">".join(ph)

        zmu0,zv0=seg[0]["zmu"],seg[0]["zv"]
        zmu1,zv1=seg[-1]["zmu"],seg[-1]["zv"]
        net_mu=zmu1-zmu0; net_var=zv1-zv0
        path=0.0
        max_var=max(s["zv"] for s in seg)
        min_var=min(s["zv"] for s in seg)
        max_abs_mu=max(abs(s["zmu"]) for s in seg)
        for a,b in zip(seg,seg[1:]):
            path += float(np.hypot(b["zmu"]-a["zmu"], b["zv"]-a["zv"]))
        roads.append({
            "template":template,
            "steps":len(seg)-1,
            "start_round":seg[0]["round"],
            "end_round":seg[-1]["round"],
            "net_mu":net_mu,
            "net_var":net_var,
            "path_len":path,
            "max_var":max_var,
            "min_var":min_var,
            "max_abs_mu":max_abs_mu,
        })
        i=k

    tc=Counter(r["template"] for r in roads)
    by=defaultdict(list)
    for r in roads: by[r["template"]].append(r)

    template_summary={}
    for t,rr in by.items():
        template_summary[t]={
            "count":len(rr),
            "share":len(rr)/max(1,len(roads)),
            "mean_steps":float(np.mean([x["steps"] for x in rr])),
            "mean_net_mu":float(np.mean([x["net_mu"] for x in rr])),
            "mean_net_var":float(np.mean([x["net_var"] for x in rr])),
            "mean_path_len":float(np.mean([x["path_len"] for x in rr])),
            "mean_max_abs_mu":float(np.mean([x["max_abs_mu"] for x in rr])),
        }

    all_steps=np.array([r["steps"] for r in roads],dtype=float) if roads else np.array([])
    all_path=np.array([r["path_len"] for r in roads],dtype=float) if roads else np.array([])
    all_var=np.array([r["net_var"] for r in roads],dtype=float) if roads else np.array([])
    return {
        "n_roads":len(roads),
        "template_summary":template_summary,
        "top_templates":[x for x,_ in tc.most_common(20)],
        "road_steps_mean":float(all_steps.mean()) if len(all_steps) else None,
        "road_steps_median":float(np.median(all_steps)) if len(all_steps) else None,
        "road_path_mean":float(all_path.mean()) if len(all_path) else None,
        "road_net_var_mean":float(all_var.mean()) if len(all_var) else None,
    }


def compare(actual,worlds):
    keys=set(actual["template_summary"])
    for w in worlds: keys.update(w["template_summary"])
    rows=[]
    for t in keys:
        a=actual["template_summary"].get(t,{"share":0,"count":0})
        vals=np.asarray([w["template_summary"].get(t,{"share":0})["share"] for w in worlds],dtype=float)
        mean=float(vals.mean()); sd=float(vals.std(ddof=1))
        z=(a["share"]-mean)/sd if sd>1e-12 else 0.0
        rows.append({
            "template":t,
            "actual_count":int(a["count"]),
            "actual_share":float(a["share"]),
            "null_mean_share":mean,
            "delta":float(a["share"]-mean),
            "z":float(z),
            "percentile":float(np.mean(vals<=a["share"]))
        })
    rows.sort(key=lambda x:abs(x["z"]),reverse=True)

    scalars={}
    for fld in ["n_roads","road_steps_mean","road_steps_median","road_path_mean","road_net_var_mean"]:
        vals=np.asarray([w[fld] for w in worlds],dtype=float)
        a=float(actual[fld])
        scalars[fld]={
            "actual":a,"null_mean":float(vals.mean()),
            "q025":float(np.quantile(vals,.025)),"q975":float(np.quantile(vals,.975)),
            "percentile":float(np.mean(vals<=a))
        }
    return {"scalar":scalars,"top_template_residuals":rows[:24]}


def main():
    rows=fetch_history()
    rounds=[r for r,_ in rows]
    actual_draws=np.asarray([x for _,x in rows],dtype=int)

    rng=np.random.default_rng(SEED)
    cal=calibrate(rng)
    actual=evaluate(actual_draws,cal,rounds)
    worlds=[evaluate(random_world(rng,len(rows)),cal) for _ in range(NULL_WORLDS)]

    out={
        "probe":"LOTO6 Flow-to-Flow Roads v0",
        "source":URL,
        "rounds":[rounds[0],rounds[-1]],
        "draws":len(rows),
        "definition":{
            "road":"last Flow state before leaving Flow -> first later Flow state",
            "phase_codes":{"F":"Flow","B":"Break","U":"Uncertainty","R":"Re-formation"},
            "template":"consecutive duplicate phases compressed; e.g. F>B>R>F",
            "shape_path":"centroid x variance path in B-standardized coordinates",
        },
        "B_calibration":cal,
        "actual":actual,
        "comparison":compare(actual,worlds),
        "null":{"worlds":NULL_WORLDS,"seed":SEED},
        "boundary":[
            "Exploratory large-map observation; road templates are descriptive.",
            "No route is selected for prediction here.",
            "Rare templates can look extreme by chance; consistency checks come later."
        ]
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
