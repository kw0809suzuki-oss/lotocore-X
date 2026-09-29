from __future__ import annotations

from collections import Counter
from pathlib import Path
import statistics

import pandas as pd

import probe_loto7_dynamic_band_online_replay_v0 as box

DATA = Path("data/loto7.csv")
WINDOW = box.WINDOW
K_NEXT = 5
MIN_SAME_STATE = 8


def state_of(snap):
    # Simple A-world state: center tercile x span tercile, computed from prior-only history.
    return {"center": snap["center"], "span": snap["span"]}


def q(values, p):
    xs = sorted(values)
    if not xs:
        return 0.0
    pos = (len(xs)-1)*p
    lo = int(pos)
    hi = min(lo+1, len(xs)-1)
    w = pos-lo
    return xs[lo]*(1-w)+xs[hi]*w


def classify(cur, prior_states):
    centers=[s["center"] for s in prior_states]
    spans=[s["span"] for s in prior_states]
    cq1,cq2=q(centers,1/3),q(centers,2/3)
    sq1,sq2=q(spans,1/3),q(spans,2/3)
    ci=0 if cur["center"]<cq1 else (2 if cur["center"]>cq2 else 1)
    si=0 if cur["span"]<sq1 else (2 if cur["span"]>sq2 else 1)
    return (ci,si)


def hits(actual, picks):
    return len(set(actual)&set(picks))


def transition_envelope(cur_snap, state_key, historical_transitions):
    # Historical next-state destinations observed after the same pre-draw state.
    next_counts=Counter(dst for src,dst in historical_transitions if src==state_key)
    if not next_counts:
        return tuple(cur_snap["core"]), 0, []

    destinations=[k for k,_ in next_counts.most_common(K_NEXT)]

    # Map each destination to a representative target center/span from prior observations.
    reps={}
    for dst in destinations:
        reps[dst]=[]

    # historical_transitions entries may include representative next boxes as 3rd item.
    for rec in historical_transitions:
        if len(rec)==3:
            src,dst,next_snap=rec
            if src==state_key and dst in reps:
                reps[dst].append(next_snap)

    route_sets=[]
    for dst in destinations:
        samples=reps.get(dst,[])
        if not samples:
            continue
        tc=statistics.mean(s["center"] for s in samples)
        ts=statistics.mean(s["span"] for s in samples)
        cands=box.precompute(cur_snap)
        route_sets.append(box.pick(cands,tc,ts))

    if not route_sets:
        return tuple(cur_snap["core"]), sum(next_counts.values()), destinations

    # Strength = survives across several plausible next-state routes.
    freq=Counter(n for s in route_sets for n in s)
    ranked=sorted(cur_snap["support"], key=lambda n:(-freq[n],-cur_snap["scores"][n],n))
    final7=tuple(sorted(ranked[:7]))
    return final7, sum(next_counts.values()), destinations


def summarize(rows,label):
    if not rows:
        return
    base=statistics.mean(r["base_h"] for r in rows)
    env=statistics.mean(r["env_h"] for r in rows)
    p3b=statistics.mean(r["base_h"]>=3 for r in rows)
    p3e=statistics.mean(r["env_h"]>=3 for r in rows)
    p4b=statistics.mean(r["base_h"]>=4 for r in rows)
    p4e=statistics.mean(r["env_h"]>=4 for r in rows)
    print(f"--- {label} n={len(rows)} ---")
    print(f"BASE mean={base:.6f} 3+={p3b:.6%} 4+={p4b:.6%}")
    print(f"ENV  mean={env:.6f} 3+={p3e:.6%} 4+={p4e:.6%}")
    print(f"DELTA mean={env-base:+.6f} 3+={(p3e-p3b)*100:+.3f}pt 4+={(p4e-p4b)*100:+.3f}pt")
    print()


def main():
    df=pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    boxes={}
    for i in range(WINDOW,len(df)):
        boxes[int(df.iloc[i]["round"])]=box.snap(df.iloc[i-WINDOW:i])

    transitions=[]
    rows=[]
    prior_state_samples=[]

    # We require enough prior state observations to define terciles and enough same-state history.
    for i in range(WINDOW+2,len(df)):
        r=int(df.iloc[i]["round"])
        pr=int(df.iloc[i-1]["round"])
        ppr=int(df.iloc[i-2]["round"])
        cur=boxes[r]
        prev=boxes[pr]
        prevprev=boxes[ppr]
        actual=box.truth(df.iloc[i])

        # State definitions use only observations strictly before target r.
        if len(prior_state_samples)<30:
            prior_state_samples.append(state_of(prevprev))
            continue

        prev_key=classify(state_of(prev),prior_state_samples)
        prevprev_key=classify(state_of(prevprev),prior_state_samples)

        same_n=sum(1 for rec in transitions if rec[0]==prev_key)
        env7,support_n,dests=transition_envelope(cur,prev_key,transitions)
        base=tuple(cur["core"])

        if same_n>=MIN_SAME_STATE:
            rows.append({
                "round":r,
                "state":prev_key,
                "same_n":same_n,
                "dest_n":len(dests),
                "base_h":hits(actual,base),
                "env_h":hits(actual,env7),
            })

        # Only after scoring r, add the observed transition ending at current state.
        cur_key=classify(state_of(cur),prior_state_samples+[state_of(prevprev),state_of(prev)])
        transitions.append((prev_key,cur_key,cur))
        prior_state_samples.append(state_of(prevprev))

    print("=== LOTO7 TRANSITION ENVELOPE v0 ===")
    print("Idea: do not choose one B-world. Use historically observed next-state routes from the current A-state,")
    print("then rank numbers by how many plausible routes they survive.")
    print("Strict chronology: target outcome is never used to build its envelope.")
    print(f"K_NEXT={K_NEXT} MIN_SAME_STATE={MIN_SAME_STATE}")
    print()
    summarize(rows,"ALL ELIGIBLE")
    summarize(rows[-200:],"LAST200")
    summarize(rows[-100:],"LAST100")

    if rows:
        print("RECENT 20")
        for r in rows[-20:]:
            print(f"r{r['round']} state={r['state']} same_n={r['same_n']} dests={r['dest_n']} base={r['base_h']} env={r['env_h']}")


if __name__=="__main__":
    main()
