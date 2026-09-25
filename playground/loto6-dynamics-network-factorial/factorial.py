from __future__ import annotations

import csv
import io
import itertools
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import requests

URL = "https://www.mk-mode.com/rails/loto/LOTO6_ALL.csv"
OUT = Path("results/loto6_dynamics_network_factorial_v0.json")

SEED = 20260925
BURN = 150
CORRIDOR_WIDTH = 90
MIN_BRANCH = 8
FIT_NEIGHBORS = 8
POOL_SIZE = 18
TICKETS = 20
REPS = 16
RANDOM_DESIGN_SAMPLES = 300
LOCAL_SEARCH_STEPS = 1800

CENTER = 37 / 6
CENTER_BAND = 0.25
MU_MEAN = 22.0
MU_SD = 4.755114206198086
VAR_MEAN = 131.38888888573075
VAR_SD = 54.75268470147889
PHASES = ["Flow", "Break", "Uncertainty", "Re-formation"]


def fetch_history():
    r = requests.get(URL, timeout=30, headers={"User-Agent": "lotocore-X dynamics-network-factorial-v0"})
    r.raise_for_status()
    try:
        text = r.content.decode("cp932")
    except UnicodeDecodeError:
        text = r.content.decode("utf-8")
    rows = []
    rd = csv.reader(io.StringIO(text))
    next(rd, None)
    for row in rd:
        try:
            rnd = int(row[0])
            nums = [int(row[2 + i]) for i in range(6)]
        except Exception:
            continue
        if len(nums) == 6 and len(set(nums)) == 6 and all(1 <= n <= 43 for n in nums):
            rows.append((rnd, sorted(nums)))
    rows.sort(key=lambda x: x[0])
    if len(rows) < 1000:
        raise RuntimeError(f"too few rows: {len(rows)}")
    return rows


def random_world(rng, n):
    u = rng.random((n, 43))
    idx = np.argpartition(u, 6, axis=1)[:, :6] + 1
    return np.sort(idx, axis=1)


def draw_shape(draw):
    a = np.asarray(draw, dtype=float)
    return float(a.mean()), float(a.var())


def classify_phase(A, prev_A, prev_label):
    x = A - CENTER
    if abs(x) <= CENTER_BAND:
        return "Uncertainty"
    dA = 0.0 if prev_A is None else A - prev_A
    centerward = (x * dA) < 0
    if centerward and prev_label in {"Break", "Uncertainty"}:
        return "Re-formation"
    if centerward:
        return "Flow"
    return "Break"


def motion_family(dmu, dvar):
    am, av = abs(dmu), abs(dvar)
    t = 0.35
    if am <= t and av <= t:
        return "Hold"
    if av > am * 1.25:
        return "Expand" if dvar > 0 else "Contract"
    if am > av * 1.25:
        return "Shift-R" if dmu > 0 else "Shift-L"
    return "Shift+Expand" if dvar > 0 else "Shift+Contract"


def build_states(draws):
    draws = np.asarray(draws, dtype=int)
    shapes = [draw_shape(d) for d in draws]
    ages = np.zeros(43, dtype=float)
    prev_A = None
    prev_phase = "Uncertainty"
    states = []

    for i, draw in enumerate(draws):
        A = float(ages.mean())
        phase = classify_phase(A, prev_A, prev_phase)
        if i == 0:
            zmu, zv = 0.0, 0.0
        else:
            mu, vv = shapes[i - 1]
            zmu = (mu - MU_MEAN) / MU_SD
            zv = (vv - VAR_MEAN) / VAR_SD
        dA = 0.0 if prev_A is None else A - prev_A
        if states:
            dzmu = zmu - states[-1]["zmu"]
            dzv = zv - states[-1]["zv"]
            pphase = states[-1]["phase"]
        else:
            dzmu = dzv = 0.0
            pphase = "None"
        states.append({
            "A": A,
            "dA": dA,
            "phase": phase,
            "prev_phase": pphase,
            "zmu": zmu,
            "zv": zv,
            "dzmu": dzmu,
            "dzv": dzv,
            "motion": motion_family(dzmu, dzv) if i else "None",
        })
        ages += 1
        ages[np.asarray(draw, dtype=int) - 1] = 0
        prev_A = A
        prev_phase = phase

    A = float(ages.mean())
    phase = classify_phase(A, prev_A, prev_phase)
    mu, vv = shapes[-1]
    zmu = (mu - MU_MEAN) / MU_SD
    zv = (vv - VAR_MEAN) / VAR_SD
    dA = A - prev_A
    dzmu = zmu - states[-1]["zmu"]
    dzv = zv - states[-1]["zv"]
    next_state = {
        "A": A,
        "dA": dA,
        "phase": phase,
        "prev_phase": states[-1]["phase"],
        "zmu": zmu,
        "zv": zv,
        "dzmu": dzmu,
        "dzv": dzv,
        "motion": motion_family(dzmu, dzv),
    }
    return states, next_state


def oriented_age_velocity(states, t):
    if t <= 0:
        return 0.0
    prev_x = states[t - 1]["A"] - CENTER
    return -np.sign(prev_x) * states[t]["dA"]


def corridor_signature(states, i):
    vals = []
    for lag in (2, 1, 0):
        t = i - lag
        if t <= 0:
            vals.extend([0.0, 0.0, 0.0])
        else:
            vals.extend([
                float(oriented_age_velocity(states, t)),
                float(states[t]["dzmu"]),
                float(states[t]["dzv"]),
            ])
    return np.asarray(vals, dtype=float)


def calibrate_signature_scale(rng):
    world = random_world(rng, 30000)
    states, _ = build_states(world)
    X = np.asarray([corridor_signature(states, i) for i in range(500, len(states))], dtype=float)
    mean = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    sd[sd <= 1e-12] = 1.0
    return mean, sd


def signature_matrix(states, mean, sd):
    return np.asarray([(corridor_signature(states, i) - mean) / sd for i in range(len(states))], dtype=float)


def corridor_with_distances(states, X, i):
    js = np.asarray([j for j in range(3, i) if states[j]["phase"] == states[i]["phase"]], dtype=int)
    if len(js) == 0:
        return [], {}
    d = np.sqrt(np.sum((X[js] - X[i]) ** 2, axis=1))
    k = min(CORRIDOR_WIDTH, len(js))
    order = np.argsort(d, kind="stable")[:k]
    selected = [int(js[x]) for x in order]
    dist = {int(js[x]): float(d[x]) for x in order}
    return selected, dist


def branch_fit(neigh, dist, states, road):
    ds = sorted(dist[j] for j in neigh if states[j + 1]["phase"] == road)
    if len(ds) < MIN_BRANCH:
        return None
    m = min(FIT_NEIGHBORS, len(ds))
    return -float(np.mean(ds[:m]))


def counts_for_road(draws, neigh, states, road):
    js = [j for j in neigh if states[j + 1]["phase"] == road]
    counts = np.zeros(43, dtype=float)
    for j in js:
        for n in draws[j]:
            counts[int(n) - 1] += 1
    return js, counts


def ranked_numbers(counts, salt):
    rng = np.random.default_rng(SEED + 1000003 * (salt + 1))
    jitter = rng.random(43) * 1e-9
    order = np.argsort(-(counts + jitter), kind="stable")
    return [int(x + 1) for x in order]


def build_dynamics_records(draws, mean, sd):
    draws = np.asarray(draws, dtype=int)
    states0, next_state = build_states(draws)
    states = states0 + [next_state]
    X = signature_matrix(states, mean, sd)
    records = []
    for i in range(BURN, len(draws)):
        neigh, dist = corridor_with_distances(states, X, i)
        if not neigh:
            continue
        fits = {}
        for road in PHASES:
            f = branch_fit(neigh, dist, states, road)
            if f is not None:
                fits[road] = f
        actual_road = states[i + 1]["phase"]
        if actual_road not in fits or len(fits) < 2:
            continue
        wrong = [v for road, v in fits.items() if road != actual_road]
        fit_margin = fits[actual_road] - float(np.mean(wrong))
        js, counts = counts_for_road(draws, neigh, states, actual_road)
        ranking = ranked_numbers(counts, i)
        records.append({
            "index": i,
            "actual_road": actual_road,
            "fit_margin": fit_margin,
            "branch_examples": len(js),
            "candidate18": ranking[:POOL_SIZE],
        })
    return records


def triple_index_map(n=POOL_SIZE):
    triples = list(itertools.combinations(range(n), 3))
    return triples, {t: i for i, t in enumerate(triples)}


def ticket_triple_mask(ticket, triple_to_idx):
    mask = 0
    for tri in itertools.combinations(ticket, 3):
        mask |= 1 << triple_to_idx[tuple(sorted(tri))]
    return mask


def build_abstract_ticket_space():
    triples, triple_to_idx = triple_index_map()
    tickets = list(itertools.combinations(range(POOL_SIZE), 6))
    masks = [ticket_triple_mask(t, triple_to_idx) for t in tickets]
    return triples, triple_to_idx, tickets, masks


def subset_triple_masks(triple_to_idx):
    by_k = {}
    for k in range(3, 7):
        masks = []
        for subset in itertools.combinations(range(POOL_SIZE), k):
            mask = 0
            for tri in itertools.combinations(subset, 3):
                mask |= 1 << triple_to_idx[tuple(sorted(tri))]
            masks.append(mask)
        by_k[k] = masks
    return by_k


def exact_global_coverage_count(union_triples, subset_masks):
    total = 0
    outside = 43 - POOL_SIZE
    for k in range(3, 7):
        weight = math.comb(outside, 6 - k)
        covered_internal = sum(1 for sm in subset_masks[k] if sm & union_triples)
        total += covered_internal * weight
    return total


def union_mask(indices, masks):
    u = 0
    for idx in indices:
        u |= masks[idx]
    return u


def build_network_design(tickets, masks, subset_masks, rng):
    # Stage 1: maximize distinct 3-number relations.
    selected = []
    selected_set = set()
    used = 0
    number_load = np.zeros(POOL_SIZE, dtype=int)
    for _ in range(TICKETS):
        best = None
        best_key = None
        for idx, ticket in enumerate(tickets):
            if idx in selected_set:
                continue
            gain = (masks[idx] & ~used).bit_count()
            projected = number_load.copy()
            projected[list(ticket)] += 1
            spread_penalty = int(projected.max() - projected.min())
            sq = int(np.dot(projected, projected))
            key = (gain, -spread_penalty, -sq, -idx)
            if best_key is None or key > best_key:
                best_key = key
                best = idx
        selected.append(best)
        selected_set.add(best)
        used |= masks[best]
        number_load[list(tickets[best])] += 1

    best_score = exact_global_coverage_count(union_mask(selected, masks), subset_masks)

    # Stage 2: target-blind local search on exact uniform 6-of-43 coverage.
    for _ in range(LOCAL_SEARCH_STEPS):
        pos = int(rng.integers(0, TICKETS))
        cand = int(rng.integers(0, len(tickets)))
        if cand in selected_set:
            continue
        trial = selected.copy()
        old = trial[pos]
        trial[pos] = cand
        score = exact_global_coverage_count(union_mask(trial, masks), subset_masks)
        if score > best_score:
            selected = trial
            selected_set.remove(old)
            selected_set.add(cand)
            best_score = score

    return selected, best_score, number_load


def random_design_stats(tickets, masks, subset_masks, rng):
    scores = []
    unique_triples = []
    for _ in range(RANDOM_DESIGN_SAMPLES):
        idxs = rng.choice(len(tickets), size=TICKETS, replace=False)
        u = union_mask([int(x) for x in idxs], masks)
        scores.append(exact_global_coverage_count(u, subset_masks))
        unique_triples.append(u.bit_count())
    arr = np.asarray(scores, dtype=float)
    return {
        "samples": RANDOM_DESIGN_SAMPLES,
        "coverage_mean": float(arr.mean() / math.comb(43, 6)),
        "coverage_q025": float(np.quantile(arr, 0.025) / math.comb(43, 6)),
        "coverage_q50": float(np.quantile(arr, 0.5) / math.comb(43, 6)),
        "coverage_q975": float(np.quantile(arr, 0.975) / math.comb(43, 6)),
        "unique_triples_mean": float(np.mean(unique_triples)),
    }


def map_design_to_pool(design_indices, abstract_tickets, pool, rng):
    perm = list(np.asarray(pool, dtype=int)[rng.permutation(len(pool))])
    return [[int(perm[p]) for p in abstract_tickets[idx]] for idx in design_indices]


def random_bundle_from_pool(abstract_tickets, pool, rng):
    perm = list(np.asarray(pool, dtype=int)[rng.permutation(len(pool))])
    idxs = rng.choice(len(abstract_tickets), size=TICKETS, replace=False)
    return [[int(perm[p]) for p in abstract_tickets[int(idx)]] for idx in idxs]


def pure_random_bundle(rng):
    out = []
    seen = set()
    while len(out) < TICKETS:
        t = tuple(sorted(int(x) for x in rng.choice(np.arange(1, 44), size=6, replace=False)))
        if t not in seen:
            seen.add(t)
            out.append(list(t))
    return out


def bundle_max_hit(bundle, target):
    target = set(int(x) for x in target)
    return max(len(target.intersection(ticket)) for ticket in bundle)


def summarize_hits(values):
    a = np.asarray(values, dtype=float)
    return {
        "n": int(len(a)),
        "mean_max_hits": float(a.mean()),
        "p_ge_3": float(np.mean(a >= 3)),
        "p_ge_4": float(np.mean(a >= 4)),
        "p_ge_5": float(np.mean(a >= 5)),
        "max_hit_distribution": {str(k): int(np.sum(a == k)) for k in range(7)},
    }


def summarize_pool_hits(values):
    a = np.asarray(values, dtype=float)
    return {
        "n": int(len(a)),
        "mean_pool_hits": float(a.mean()),
        "p_pool_ge_3": float(np.mean(a >= 3)),
        "p_pool_ge_4": float(np.mean(a >= 4)),
        "p_pool_ge_5": float(np.mean(a >= 5)),
    }


def random_pool_exact():
    den = math.comb(43, 6)
    pmf = {
        k: math.comb(POOL_SIZE, k) * math.comb(43 - POOL_SIZE, 6 - k) / den
        for k in range(7)
        if k <= POOL_SIZE and 0 <= 6 - k <= 43 - POOL_SIZE
    }
    return {
        "mean_pool_hits": 6 * POOL_SIZE / 43,
        "p_pool_ge_3": sum(v for k, v in pmf.items() if k >= 3),
        "p_pool_ge_4": sum(v for k, v in pmf.items() if k >= 4),
        "p_pool_ge_5": sum(v for k, v in pmf.items() if k >= 5),
    }


def pure_random20_expected_p3():
    p_single = sum(
        math.comb(6, k) * math.comb(37, 6 - k) / math.comb(43, 6)
        for k in range(3, 7)
    )
    return 1 - (1 - p_single) ** TICKETS


def main():
    rows = fetch_history()
    rounds = [r for r, _ in rows]
    draws = np.asarray([d for _, d in rows], dtype=int)

    calibration_rng = np.random.default_rng(SEED)
    sig_mean, sig_sd = calibrate_signature_scale(calibration_rng)
    records = build_dynamics_records(draws, sig_mean, sig_sd)

    triples, triple_to_idx, abstract_tickets, ticket_masks = build_abstract_ticket_space()
    subset_masks = subset_triple_masks(triple_to_idx)

    design_rng = np.random.default_rng(SEED + 11)
    network_design, network_covered, _ = build_network_design(
        abstract_tickets, ticket_masks, subset_masks, design_rng
    )
    network_union = union_mask(network_design, ticket_masks)
    network_theory = {
        "pool_size": POOL_SIZE,
        "tickets": TICKETS,
        "unique_triples": int(network_union.bit_count()),
        "exact_global_p_ge_3": float(network_covered / math.comb(43, 6)),
        "pool_hit_ceiling_p_ge_3": float(random_pool_exact()["p_pool_ge_3"]),
        "ceiling_efficiency": float(
            (network_covered / math.comb(43, 6)) / random_pool_exact()["p_pool_ge_3"]
        ),
        "abstract_design": [list(map(int, abstract_tickets[idx])) for idx in network_design],
    }
    random_alloc_theory = random_design_stats(
        abstract_tickets, ticket_masks, subset_masks, np.random.default_rng(SEED + 22)
    )

    cells = {
        "random18_random20": [],
        "random18_network20": [],
        "dynamics18_random20": [],
        "dynamics18_network20": [],
        "pure_random20": [],
    }
    random_pool_hits = []
    dynamics_pool_hits = []

    # Paired values for effect decomposition at the same target/rep.
    effects = {
        "max_hits": {"dynamics": [], "network": [], "combined": [], "interaction": []},
        "ge3": {"dynamics": [], "network": [], "combined": [], "interaction": []},
        "ge4": {"dynamics": [], "network": [], "combined": [], "interaction": []},
    }

    eval_rng = np.random.default_rng(SEED + 33)
    for rec in records:
        i = rec["index"]
        target = draws[i]
        dyn_pool = list(rec["candidate18"])
        dyn_pool_hit = len(set(dyn_pool).intersection(int(x) for x in target))

        for _ in range(REPS):
            rnd_pool = sorted(int(x) for x in eval_rng.choice(np.arange(1, 44), size=POOL_SIZE, replace=False))
            rnd_pool_hit = len(set(rnd_pool).intersection(int(x) for x in target))
            random_pool_hits.append(rnd_pool_hit)
            dynamics_pool_hits.append(dyn_pool_hit)

            rr = bundle_max_hit(random_bundle_from_pool(abstract_tickets, rnd_pool, eval_rng), target)
            rn = bundle_max_hit(map_design_to_pool(network_design, abstract_tickets, rnd_pool, eval_rng), target)
            dr = bundle_max_hit(random_bundle_from_pool(abstract_tickets, dyn_pool, eval_rng), target)
            dn = bundle_max_hit(map_design_to_pool(network_design, abstract_tickets, dyn_pool, eval_rng), target)
            pr = bundle_max_hit(pure_random_bundle(eval_rng), target)

            cells["random18_random20"].append(rr)
            cells["random18_network20"].append(rn)
            cells["dynamics18_random20"].append(dr)
            cells["dynamics18_network20"].append(dn)
            cells["pure_random20"].append(pr)

            for metric, transform in {
                "max_hits": lambda x: float(x),
                "ge3": lambda x: float(x >= 3),
                "ge4": lambda x: float(x >= 4),
            }.items():
                base = transform(rr)
                dyn = transform(dr)
                net = transform(rn)
                combo = transform(dn)
                effects[metric]["dynamics"].append(dyn - base)
                effects[metric]["network"].append(net - base)
                effects[metric]["combined"].append(combo - base)
                effects[metric]["interaction"].append(combo - dyn - net + base)

    effect_summary = {}
    for metric, parts in effects.items():
        effect_summary[metric] = {
            key: float(np.mean(vals)) for key, vals in parts.items()
        }

    out = {
        "probe": "LOTO6 Dynamics × Network Factorial v0",
        "source": URL,
        "rounds": [int(rounds[0]), int(rounds[-1])],
        "draws": len(rows),
        "historical_records": len(records),
        "reps_per_record": REPS,
        "question": "Does a target-seeking dynamics selector and a target-blind spread network allocator compose usefully, and is there interaction beyond their separate effects?",
        "origin_split": {
            "dynamics": "Aim: choose an 18-number material region from the Route Corridor model. Historical diagnostic conditions on the actually realized next road.",
            "network": "Spread: with the material pool fixed, arrange the same 20 tickets to maximize exact uniform-world 3+ coverage. It never sees the target draw or route score.",
            "bridge": "Dynamics chooses WHERE to place material; Network chooses HOW to arrange 20 tickets inside that material.",
        },
        "factorial": {
            "selector_off": "uniform random 18-number pool",
            "selector_on": "correct-road Route Corridor Candidate18 (oracle-conditioned historical diagnostic)",
            "allocator_off": "20 distinct random 6-number tickets inside the selected 18-number pool",
            "allocator_on": "one frozen target-blind 20-ticket network design optimized for exact 3+ coverage inside an abstract 18-number pool",
            "paired_repetitions": REPS,
        },
        "selector_layer": {
            "random18_exact": random_pool_exact(),
            "random18_observed_control": summarize_pool_hits(random_pool_hits),
            "dynamics18_observed": summarize_pool_hits(dynamics_pool_hits),
        },
        "network_layer": {
            "network20": network_theory,
            "random20_within18": random_alloc_theory,
            "pure_random20_global_independent_reference_p_ge_3": pure_random20_expected_p3(),
        },
        "final_20ticket_cells": {key: summarize_hits(vals) for key, vals in cells.items()},
        "paired_effect_decomposition": {
            "definition": {
                "dynamics": "Dynamics18+Random20 minus Random18+Random20",
                "network": "Random18+Network20 minus Random18+Random20",
                "combined": "Dynamics18+Network20 minus Random18+Random20",
                "interaction": "Combined - Dynamics-only - Network-only + baseline; positive means the network helps dynamics material more than it helps random material.",
            },
            "effects": effect_summary,
        },
        "boundary": [
            "The network allocator is target-blind and uses only combinatorial coverage; its theoretical coverage is exact under uniform 6-of-43.",
            "The historical dynamics selector is oracle-conditioned on the actually realized road. This isolates the composition mechanism but does not measure a user's probability of choosing the correct road.",
            "Random18 is used as a concentration-matched selector control. Pure Random20 over all 43 is also simulated as an external practical reference.",
            "All cells use exactly 20 tickets.",
            "No Floot app changes are made by this probe.",
            "Historical A is exploratory and not a pristine future holdout.",
        ],
    }

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
