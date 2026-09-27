from __future__ import annotations

from dataclasses import dataclass
import math
import pandas as pd
import lotocore

BOUNDARY_K=6

@dataclass
class Prediction:
    numbers: tuple[int,...]
    state: dict


def _ranked(snap):
    ranks={int(n):int(r) for n,r in snap["ranks"].items()}
    return sorted(ranks,key=lambda n:(ranks[n],n))


def _weighted_state(scores):
    vals={int(n):max(0.0,float(v)) for n,v in scores.items()}
    total=sum(vals.values())
    p={n:vals.get(n,0.0)/total for n in range(1,38)}
    c=sum(n*p[n] for n in p)
    v=sum((n-c)**2*p[n] for n in p)
    return c,math.sqrt(v)


def _set_state(nums):
    vals=list(nums)
    c=sum(vals)/len(vals)
    v=sum((n-c)**2 for n in vals)/len(vals)
    return c,math.sqrt(v)


def predict(history:pd.DataFrame)->Prediction:
    base=lotocore.predict(history)
    snap=lotocore.score_snapshot(history)
    core=tuple(sorted(base.numbers))
    core_set=set(core)
    order=_ranked(snap)
    boundary=tuple(n for n in order if n not in core_set)[:BOUNDARY_K]
    scores={int(n):float(v) for n,v in snap["scores"].items()}
    tc,ts=_weighted_state(snap["scores"])

    def eval_box(box):
        c,s=_set_state(box)
        ce=abs(c-tc); se=abs(s-ts)
        err=(ce+se)/max(ts,1e-12)
        mass=sum(scores[n] for n in box)
        return err,mass,c,s,ce,se

    base_eval=eval_box(core)
    candidates=[(base_eval[0],-base_eval[1],0,0,core,base_eval)]
    for out_n in core:
        for in_n in boundary:
            cand=tuple(sorted((core_set-{out_n})|{in_n}))
            ev=eval_box(cand)
            candidates.append((ev[0],-ev[1],out_n,in_n,cand,ev))

    best=min(candidates,key=lambda x:(x[0],x[1],x[2],x[3],x[4]))
    _,_,out_n,in_n,chosen,ev=best
    err,mass,c,s,ce,se=ev
    return Prediction(chosen,{
        "model":"lotocore_finite_generation_minimal_v1",
        "base_core7":list(core),
        "boundary6":list(boundary),
        "changed":int(chosen!=core),
        "out":None if chosen==core else out_n,
        "in":None if chosen==core else in_n,
        "target_center":round(tc,6),
        "target_spread":round(ts,6),
        "center":round(c,6),
        "spread":round(s,6),
        "center_error":round(ce,6),
        "spread_error":round(se,6),
        "score_mass":round(mass,6),
    })
