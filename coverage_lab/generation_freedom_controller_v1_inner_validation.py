#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lotocore
from coverage_lab.loto7_portfolio20_v0 import (
    B_CORE, B_DIVERSIFY, MODEL_WINDOW, POOL_K,
    avg_candidate_score, euclid, main_numbers, max_overlap,
    portfolio_bundle, sampled_combinations, standardized_vectors,
    unique_structured_bundle,
)

DATA = Path("coverage_lab/fixtures/loto7_1_696.csv")
OUT = Path("results/generation_freedom_controller_v1_inner_validation.json")
LOST5 = {314, 359, 374, 413, 440}


def prep(pool, scores, seed):
    core, _ = unique_structured_bundle(pool, B_CORE, seed)
    combos = sampled_combinations(pool, seed + 5000, core)
    z = standardized_vectors(combos)
    m = {c: avg_candidate_score(c, scores) for c in combos}
    median = sorted(m.values())[len(m)//2]
    high = [c for c in combos if c not in set(core) and m[c] >= median]
    centroid = tuple(sum(z[t][i] for t in core)/len(core) for i in range(len(z[core[0]])))
    d = {c: euclid(z[c], centroid) for c in combos}
    core_min = {c: min(euclid(z[c], z[s]) for s in core) for c in high}
    return core, combos, z, m, median, high, d, core_min


def det_key(c):
    return tuple(-n for n in c)


def select_a_centroid(high, d):
    return sorted(high, key=lambda c:(d[c], det_key(c)), reverse=True)[:B_DIVERSIFY]


def select_b_core_static(high, core_min):
    return sorted(high, key=lambda c:(core_min[c], det_key(c)), reverse=True)[:B_DIVERSIFY]


def select_c_dynamic(high, core, z):
    selected = list(core)
    pool = list(high)
    out = []
    for _ in range(B_DIVERSIFY):
        min_dist = {c:min(euclid(z[c], z[s]) for s in selected) for c in pool}
        best = max(pool, key=lambda c:(min_dist[c], det_key(c)))
        out.append(best)
        selected.append(best)
        pool.remove(best)
    return out


def select_d_controller_v1(high, core, z, m):
    selected = list(core)
    pool = list(high)
    out = []
    for _ in range(B_DIVERSIFY):
        min_dist = {c:min(euclid(z[c], z[s]) for s in selected) for c in pool}
        overlap = {c:max_overlap(c, selected) for c in pool}
        best = max(pool, key=lambda c:(min_dist[c], -overlap[c], m[c], det_key(c)))
        out.append(best)
        selected.append(best)
        pool.remove(best)
    return out


def success5(tickets, actual):
    return any(len(set(t) & actual) >= 5 for t in tickets)


def jaccard(a,b):
    sa,sb=set(a),set(b)
    return len(sa&sb)/len(sa|sb)


def main():
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    rows=[]
    fidelity_mismatch=[]

    for idx in range(MODEL_WINDOW, len(df)):
        hist=df.iloc[idx-MODEL_WINDOW:idx]
        target=df.iloc[idx]
        rnd=int(target["round"])
        actual=main_numbers(target)

        snap=lotocore.score_snapshot(hist)
        ranks={int(k):int(v) for k,v in snap["ranks"].items()}
        scores={int(k):float(v) for k,v in snap["scores"].items()}
        ranked=sorted(range(1,38), key=lambda n:(ranks[n],n))
        pool=ranked[:POOL_K]
        seed=rnd*100_003+20261002

        core, combos, z, m, median, high, d, core_min = prep(pool, scores, seed+2)
        A=select_a_centroid(high,d)
        B=select_b_core_static(high,core_min)
        C=select_c_dynamic(high,core,z)
        D=select_d_controller_v1(high,core,z,m)

        portfolio, trace=portfolio_bundle(pool,scores,seed+2)
        D_current=[tuple(x) for x in trace["diversify"]]
        if D != D_current:
            fidelity_mismatch.append(rnd)

        variants={"A_centroid":A,"B_core_static":B,"C_dynamic":C,"D_controller_v1":D}
        rec={"round":rnd}
        for name,tix in variants.items():
            rec[f"{name}_success6"]=success5(tix,actual)
            rec[f"{name}_success16"]=success5(core+tix,actual)
            rec[f"{name}_hits_lost5_winner"]=rnd in LOST5 and success5(tix,actual)
        rec["J_A_B"]=jaccard(A,B)
        rec["J_B_C"]=jaccard(B,C)
        rec["J_C_D"]=jaccard(C,D)
        rec["J_A_D"]=jaccard(A,D)
        rows.append(rec)

    if fidelity_mismatch:
        raise RuntimeError(f"Controller v1 fidelity mismatch: {fidelity_mismatch[:10]}")

    names=["A_centroid","B_core_static","C_dynamic","D_controller_v1"]
    success_sets={n:{r["round"] for r in rows if r[f"{n}_success16"]} for n in names}
    success6_sets={n:{r["round"] for r in rows if r[f"{n}_success6"]} for n in names}

    windows={"W1":(101,299),"W2":(300,498),"W3":(499,696)}
    temporal={}
    for w,(lo,hi) in windows.items():
        temporal[w]={}
        for n in names:
            temporal[w][n]={
                "success16":sum(lo<=x<=hi for x in success_sets[n]),
                "success6":sum(lo<=x<=hi for x in success6_sets[n]),
            }

    pairwise={}
    for i,a in enumerate(names):
        for b in names[i+1:]:
            pairwise[f"{a}__vs__{b}"]={
                "a_only16":sorted(success_sets[a]-success_sets[b]),
                "b_only16":sorted(success_sets[b]-success_sets[a]),
                "both16":sorted(success_sets[a]&success_sets[b]),
                "union16":len(success_sets[a]|success_sets[b]),
            }

    payload={
        "experiment":"generation_freedom_controller_v1_inner_validation",
        "gate0_fidelity":{
            "rounds":len(rows),
            "controller_v1_exact_matches_current_diversify":len(rows),
            "mismatches":fidelity_mismatch,
        },
        "variants":{
            "A_centroid":"m>=round median; top6 by fixed Core-centroid d",
            "B_core_static":"m>=round median; top6 by fixed min distance to Core10, selected in one batch",
            "C_dynamic":"m>=round median; iterative min_dist to current selected set; deterministic tie only",
            "D_controller_v1":"m>=round median; iterative min_dist; lower overlap, higher m, deterministic tie-break",
        },
        "summary":{
            n:{
                "success16_count":len(success_sets[n]),
                "success6_count":len(success6_sets[n]),
                "lost5_recovered_by_diversify":sorted(success6_sets[n]&LOST5),
            } for n in names
        },
        "mean_jaccard":{
            k:sum(r[k] for r in rows)/len(rows)
            for k in ("J_A_B","J_B_C","J_C_D","J_A_D")
        },
        "temporal_transfer_inside_same_history":temporal,
        "pairwise_success_sets":pairwise,
        "boundary":[
            "This is an internal mechanism decomposition on rounds 101-696, not independent external validation.",
            "All variants share the same round, CORE18, 2500 candidates, seed, Core10, m threshold, and future-information boundary.",
            "The primary isolated outcome is Core10+Diversify6 (16 tickets); this avoids downstream Falsify differences obscuring the Diversify mechanism.",
            "D must exactly reproduce current Portfolio Diversify on all 596 rounds or the run fails.",
            "No parameter search is performed."
        ],
        "rows":rows,
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("=== GENERATION FREEDOM CONTROLLER v1 INNER VALIDATION ===")
    print("FIDELITY",json.dumps(payload["gate0_fidelity"],ensure_ascii=False,sort_keys=True))
    print("SUMMARY",json.dumps(payload["summary"],ensure_ascii=False,sort_keys=True))
    print("JACCARD",json.dumps(payload["mean_jaccard"],ensure_ascii=False,sort_keys=True))
    print("TEMPORAL",json.dumps(payload["temporal_transfer_inside_same_history"],ensure_ascii=False,sort_keys=True))
    print("saved ->",OUT)


if __name__=="__main__":
    main()
