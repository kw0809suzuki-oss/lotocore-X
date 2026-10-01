from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd
import lotocore

from probe_loto7_transition_branch10_v0 import state, transition, allocate, max_match
from compare_loto7_branch10_core10_random10 import make_structured10, core18, random_bundle

DATA = Path("data/loto7.csv")
OUT_CSV = Path("results/loto7_branch_core_hybrid_v0.csv")
OUT_JSON = Path("results/loto7_branch_core_hybrid_v0_summary.json")
RANDOM_REPS = 1000
MIN_MATCHES = 10

def branch_context(prior):
    states = [state(d["nums"]) for d in prior]
    trs = [transition(states[i], states[i+1]) for i in range(len(states)-1)]
    if len(trs) < 2:
        return None
    orbit = (trs[-2], trs[-1])
    examples = []
    for i in range(1, len(trs)-1):
        if trs[i-1] == orbit[0] and trs[i] == orbit[1]:
            examples.append({
                "branch": trs[i+1],
                "ticket": sorted(prior[i+2]["nums"]),
                "round": prior[i+2]["round"],
            })
    if len(examples) < MIN_MATCHES:
        return None
    counts = Counter(e["branch"] for e in examples)
    return orbit, examples, allocate(counts, 10)

def generalized_structured(pool_order, slots, seed):
    """CORE-like compression for an arbitrary branch slot count.

    Branch fixes how many tickets this sub-world receives.
    Within the branch, candidates are ordered by CORE rank after branch support filtering.
    Ticket construction then balances reuse and pair repetition.
    """
    if slots <= 0:
        return []
    if len(pool_order) < 7:
        return []

    total_slots = slots * 7
    # Candidate pool size scales with branch capacity, capped at 18.
    k = min(len(pool_order), max(7, min(18, total_slots)))
    pool = pool_order[:k]

    # Degree allocation: one touch first, then diminishing-return rank weight.
    deg = Counter()
    first_touch = min(k, total_slots)
    for n in pool[:first_touch]:
        deg[n] += 1
    remaining = total_slots - first_touch
    rank = {n: i for i, n in enumerate(pool)}
    while remaining > 0:
        candidates = [n for n in pool if deg[n] < slots]
        if not candidates:
            break
        pick = max(candidates, key=lambda n: ((k-rank[n])/(deg[n]+1.0), -rank[n]))
        deg[pick] += 1
        remaining -= 1

    rem = Counter(deg)
    pair_count = Counter()
    rng = random.Random(seed)
    out = []

    for i in range(slots):
        tickets_left = slots - i
        chosen = [n for n,c in rem.items() if c == tickets_left]
        if len(chosen) > 7:
            chosen = sorted(chosen, key=lambda n: rank[n])[:7]
        while len(chosen) < 7:
            candidates = [n for n,c in rem.items() if c > 0 and n not in chosen]
            if not candidates:
                return []
            rng.shuffle(candidates)
            pick = min(
                candidates,
                key=lambda n: (
                    sum(pair_count[tuple(sorted((n,x)))] for x in chosen),
                    -rem[n],
                    rank[n],
                ),
            )
            chosen.append(pick)
        t = tuple(sorted(chosen))
        out.append(list(t))
        for n in chosen:
            rem[n] -= 1
        for a in range(7):
            for b in range(a+1,7):
                pair_count[(t[a],t[b])] += 1
    return out

def hybrid10(prior, history_df, target_round):
    ctx = branch_context(prior)
    if ctx is None:
        return None
    orbit, examples, allocation = ctx

    snap = lotocore.score_snapshot(history_df.tail(100))
    core_ranks = {int(k): int(v) for k,v in snap["ranks"].items()}

    tickets = []
    details = []
    for bi, (branch, slots) in enumerate(allocation):
        support = Counter()
        for e in examples:
            if e["branch"] == branch:
                support.update(e["ticket"])
        # Keep Branch as the gate: only numbers observed in that branch's historical next tickets.
        # CORE decides priority inside that branch-supported set.
        pool_order = sorted(
            support.keys(),
            key=lambda n: (core_ranks[n], -support[n], n),
        )
        bt = generalized_structured(pool_order, slots, target_round*1009 + bi*97 + 31)
        if len(bt) != slots:
            return None
        tickets.extend(bt)
        details.append((branch, slots, len(pool_order)))
    if len(tickets) != 10:
        return None
    return {"orbit":orbit, "tickets":tickets, "allocation":allocation, "details":details}

def branch10(prior):
    ctx = branch_context(prior)
    if ctx is None:
        return None
    orbit, examples, allocation = ctx
    # nearest-history-free baseline: use most recent examples per branch to fill slots
    tickets=[]
    for branch,slots in allocation:
        pool=[e for e in examples if e["branch"]==branch]
        pool.sort(key=lambda e:e["round"], reverse=True)
        tickets.extend(e["ticket"] for e in pool[:slots])
    return {"orbit":orbit,"tickets":tickets}

def summarize(out,key):
    s=out[key]
    return {
        "mean_max_match":float(s.mean()),
        "rate_3plus":float((s>=3).mean()),
        "rate_4plus":float((s>=4).mean()),
        "rate_5plus":float((s>=5).mean()),
        "count_3plus":int((s>=3).sum()),
        "count_4plus":int((s>=4).sum()),
        "count_5plus":int((s>=5).sum()),
    }

def main():
    df=pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    cols=[f"n{i}" for i in range(1,8)]
    draws=[{"round":int(r["round"]),"nums":[int(r[c]) for c in cols]} for _,r in df.iterrows()]
    rows=[]

    for target_i in range(100,len(draws)):
        prior=draws[:target_i]
        h=hybrid10(prior,df.iloc[:target_i],draws[target_i]["round"])
        if h is None:
            continue

        # Use the original Branch10 v0 implementation for fair comparison.
        from probe_loto7_transition_branch10_v0 import build_branch10
        b=build_branch10(prior)
        if b is None:
            continue

        c18=core18(df.iloc[:target_i])
        c=make_structured10(c18,draws[target_i]["round"]*2654435761+11)
        actual=draws[target_i]["nums"]

        rng=random.Random(draws[target_i]["round"]*1000003+20261001)
        rr=[]
        for _ in range(RANDOM_REPS):
            rr.append(max_match(actual,random_bundle(rng)))

        rows.append({
            "round":draws[target_i]["round"],
            "hybrid_max_match":max_match(actual,h["tickets"]),
            "branch10_max_match":max_match(actual,b["tickets"]),
            "core10_max_match":max_match(actual,c),
            "random_mean_max_match":sum(rr)/len(rr),
            "random_rate_3plus":sum(x>=3 for x in rr)/len(rr),
            "random_rate_4plus":sum(x>=4 for x in rr)/len(rr),
            "random_rate_5plus":sum(x>=5 for x in rr)/len(rr),
            "allocation":json.dumps(h["allocation"],ensure_ascii=False),
        })

    out=pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(OUT_CSV,index=False)

    summary={
        "comparison":"Branch->CORE Hybrid10 v0 vs Branch10 vs Core10 vs Random10",
        "evaluated_rounds":len(out),
        "history_round_max":int(df["round"].max()),
        "random_reps_per_round":RANDOM_REPS,
        "hybrid10":summarize(out,"hybrid_max_match"),
        "branch10":summarize(out,"branch10_max_match"),
        "core10":summarize(out,"core10_max_match"),
        "random10":{
            "mean_max_match":float(out["random_mean_max_match"].mean()),
            "rate_3plus":float(out["random_rate_3plus"].mean()),
            "rate_4plus":float(out["random_rate_4plus"].mean()),
            "rate_5plus":float(out["random_rate_5plus"].mean()),
        },
        "paired_hybrid_vs_branch":{
            "hybrid_gt":int((out.hybrid_max_match>out.branch10_max_match).sum()),
            "equal":int((out.hybrid_max_match==out.branch10_max_match).sum()),
            "hybrid_lt":int((out.hybrid_max_match<out.branch10_max_match).sum()),
        },
        "paired_hybrid_vs_core":{
            "hybrid_gt":int((out.hybrid_max_match>out.core10_max_match).sum()),
            "equal":int((out.hybrid_max_match==out.core10_max_match).sum()),
            "hybrid_lt":int((out.hybrid_max_match<out.core10_max_match).sum()),
        },
        "definition":"Branch fixes 10-ticket allocation and branch-supported number sets; repository CORE rank orders candidates inside each branch; CORE-like degree/pair-diversity allocator compresses each branch into its assigned ticket slots.",
        "boundary":[
            "Hybrid v0 is a new candidate, not a recovered historical model.",
            "CORE side is the deterministic repository proxy, not Floot AI-generated CORE18.",
            "Branch10 comparator is historical-analog v0.",
            "Historical walk-forward comparison does not establish future lottery advantage."
        ]
    }
    OUT_JSON.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
