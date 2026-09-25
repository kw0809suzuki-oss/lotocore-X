from __future__ import annotations

import csv
import io
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import requests

URL = "https://www.mk-mode.com/rails/loto/LOTO6_ALL.csv"
OUT = Path("results/loto6_dynamics_network_coupling_v0.json")

SEED = 20260925
BURN = 150
CORRIDOR_WIDTH = 90
MIN_BRANCH = 8
FIT_NEIGHBORS = 8
POOL_SIZES = [18, 15, 12]
RETENTION = [1.00, 0.50, 0.30, 0.20, 0.10]
ALLOCATOR_SEEDS = 512
TICKETS = 20
DRAW_SIZE = 6
TOTAL_SLOTS = TICKETS * DRAW_SIZE

CENTER = 37 / 6
CENTER_BAND = 0.25
MU_MEAN = 22.0
MU_SD = 4.755114206198086
VAR_MEAN = 131.38888888573075
VAR_SD = 54.75268470147889
PHASES = ["Flow", "Break", "Uncertainty", "Re-formation"]
OUTCOME_METRICS = ["max_hit", "p3", "p4", "p5", "p6"]


def fetch_history():
    r = requests.get(URL, timeout=30, headers={"User-Agent": "lotocore-X dynamics-network-coupling-v0"})
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
    t = .35
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
            "A": A, "dA": dA, "phase": phase, "prev_phase": pphase,
            "zmu": zmu, "zv": zv, "dzmu": dzmu, "dzv": dzv,
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
        "A": A, "dA": dA, "phase": phase, "prev_phase": states[-1]["phase"],
        "zmu": zmu, "zv": zv, "dzmu": dzmu, "dzv": dzv,
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


def ranked_numbers_for_road(draws, neigh, states, road, salt):
    counts = np.zeros(43, dtype=float)
    js = [j for j in neigh if states[j + 1]["phase"] == road]
    for j in js:
        for n in draws[j]:
            counts[int(n) - 1] += 1
    rng = np.random.default_rng(SEED + 1000003 * (salt + 1))
    jitter = rng.random(43) * 1e-9
    order = np.argsort(-(counts + jitter), kind="stable")
    return [int(x + 1) for x in order], len(js)


def build_capture_records(draws, rounds, mean, sd):
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
        ranking, branch_examples = ranked_numbers_for_road(draws, neigh, states, actual_road, i)
        actual = set(int(x) for x in draws[i])
        captures = {str(p): len(set(ranking[:p]) & actual) for p in POOL_SIZES}
        records.append({
            "index": int(i),
            "round": int(rounds[i]),
            "road": actual_road,
            "fit_margin": float(fit_margin),
            "branch_examples": int(branch_examples),
            "captures": captures,
        })
    return records


def balanced_degrees(pool, seed):
    pool = list(pool)
    rng = random.Random(seed)
    shuffled = pool[:]
    rng.shuffle(shuffled)
    base, rem = divmod(TOTAL_SLOTS, len(pool))
    deg = Counter({n: base for n in pool})
    for n in shuffled[:rem]:
        deg[n] += 1
    return deg


def choose_ticket(remaining, tickets_left, rng, pair_count=None):
    chosen = [n for n, c in remaining.items() if c == tickets_left]
    if len(chosen) > DRAW_SIZE:
        raise RuntimeError("infeasible degree sequence")
    while len(chosen) < DRAW_SIZE:
        candidates = [n for n, c in remaining.items() if c > 0 and n not in chosen]
        rng.shuffle(candidates)
        if pair_count is None:
            weights = [remaining[n] for n in candidates]
            pick = rng.choices(candidates, weights=weights, k=1)[0]
        else:
            pick = min(
                candidates,
                key=lambda n: (
                    sum(pair_count[tuple(sorted((n, x)))] for x in chosen),
                    -remaining[n],
                ),
            )
        chosen.append(pick)
    for n in chosen:
        remaining[n] -= 1
    return tuple(sorted(chosen))


def random_pack(degrees, seed):
    rng = random.Random(seed)
    rem = Counter(degrees)
    out = []
    for i in range(TICKETS):
        out.append(choose_ticket(rem, TICKETS - i, rng))
    return out


def mesh_pack(degrees, seed):
    rng = random.Random(seed)
    rem = Counter(degrees)
    pair_count = defaultdict(int)
    out = []
    for i in range(TICKETS):
        t = choose_ticket(rem, TICKETS - i, rng, pair_count)
        for a, b in itertools.combinations(t, 2):
            pair_count[(a, b)] += 1
        out.append(t)
    return out


def bundle_eval(tickets, actual):
    aset = set(actual)
    hits = [len(aset & set(t)) for t in tickets]
    m = max(hits)
    pairs = Counter()
    triples = Counter()
    for t in tickets:
        pairs.update(itertools.combinations(t, 2))
        triples.update(itertools.combinations(t, 3))
    return {
        "max_hit": float(m),
        "p3": float(m >= 3),
        "p4": float(m >= 4),
        "p5": float(m >= 5),
        "p6": float(m >= 6),
        "pair_score": float(sum(v * v for v in pairs.values())),
        "unique_triples": float(len(triples)),
    }


def mean_metrics(rows):
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def allocator_response(pool_size, capture):
    pool = list(range(1, pool_size + 1))
    inside = list(range(1, capture + 1))
    outside = list(range(pool_size + 1, pool_size + 1 + (DRAW_SIZE - capture)))
    actual = inside + outside
    random_rows = []
    mesh_rows = []
    for s in range(ALLOCATOR_SEEDS):
        seed = SEED + pool_size * 100003 + capture * 1009 + s * 17
        deg = balanced_degrees(pool, seed)
        random_rows.append(bundle_eval(random_pack(deg, seed + 1), actual))
        mesh_rows.append(bundle_eval(mesh_pack(deg, seed + 1), actual))
    return {"random": mean_metrics(random_rows), "mesh": mean_metrics(mesh_rows)}


def build_allocator_table():
    table = {}
    for p in POOL_SIZES + [43]:
        table[str(p)] = {}
        captures = [6] if p == 43 else list(range(7))
        for k in captures:
            table[str(p)][str(k)] = allocator_response(p, k)
    return table


def hypergeom_capture_dist(pool_size):
    den = math.comb(43, pool_size)
    probs = {}
    for k in range(7):
        if k <= pool_size and pool_size - k <= 37:
            probs[str(k)] = math.comb(6, k) * math.comb(37, pool_size - k) / den
        else:
            probs[str(k)] = 0.0
    return probs


def empirical_capture_dist(records, p):
    counts = Counter(int(r["captures"][str(p)]) for r in records)
    return {str(k): counts.get(k, 0) / len(records) for k in range(7)}


def expected_cell(table, p, dist, allocator):
    out = {}
    for metric in table[str(p)]["0"][allocator].keys():
        out[metric] = float(sum(
            dist.get(str(k), 0.0) * table[str(p)][str(k)][allocator][metric]
            for k in range(7)
        ))
    return out


def subtract(a, b):
    return {m: float(a[m] - b[m]) for m in OUTCOME_METRICS}


def summarize_slice(records, allocator_table):
    out = {"n": len(records), "pools": {}}
    global_random = allocator_table["43"]["6"]["random"]
    global_mesh = allocator_table["43"]["6"]["mesh"]

    for p in POOL_SIZES:
        candidate_dist = empirical_capture_dist(records, p)
        matched_dist = hypergeom_capture_dist(p)
        candidate_random = expected_cell(allocator_table, p, candidate_dist, "random")
        candidate_mesh = expected_cell(allocator_table, p, candidate_dist, "mesh")
        matched_random = expected_cell(allocator_table, p, matched_dist, "random")
        matched_mesh = expected_cell(allocator_table, p, matched_dist, "mesh")

        candidate_network_gain = subtract(candidate_mesh, candidate_random)
        matched_network_gain = subtract(matched_mesh, matched_random)
        interaction = {
            m: candidate_network_gain[m] - matched_network_gain[m]
            for m in OUTCOME_METRICS
        }
        dynamics_gain = subtract(candidate_random, matched_random)
        combined_vs_global = subtract(candidate_mesh, global_random)

        captures = np.asarray([r["captures"][str(p)] for r in records], dtype=float)
        by_capture = {}
        for k in range(7):
            response = allocator_table[str(p)][str(k)]
            by_capture[str(k)] = {
                "random": response["random"],
                "mesh": response["mesh"],
                "network_gain": subtract(response["mesh"], response["random"]),
            }

        out["pools"][str(p)] = {
            "candidate_capture_mean": float(captures.mean()),
            "candidate_capture_p_ge_3": float(np.mean(captures >= 3)),
            "candidate_capture_p_ge_4": float(np.mean(captures >= 4)),
            "candidate_capture_distribution": candidate_dist,
            "size_matched_random_capture_distribution_exact": matched_dist,
            "cells": {
                "global_random": global_random,
                "global_mesh": global_mesh,
                "matched_random_random": matched_random,
                "matched_random_mesh": matched_mesh,
                "candidate_random": candidate_random,
                "candidate_mesh": candidate_mesh,
            },
            "effects": {
                "dynamics_gain_vs_size_matched_random_pool_with_random_allocator": dynamics_gain,
                "network_gain_inside_candidate_pool": candidate_network_gain,
                "network_gain_inside_size_matched_random_pool": matched_network_gain,
                "factorial_interaction_candidate_specific_network_gain": interaction,
                "combined_candidate_mesh_vs_global43_random": combined_vs_global,
            },
            "allocator_response_by_capture": by_capture,
        }
    return out


def retention_curve(records, allocator_table):
    margins = np.asarray([r["fit_margin"] for r in records], dtype=float)
    out = []
    for keep in RETENTION:
        n = max(1, int(math.ceil(len(records) * keep)))
        order = np.argsort(-margins, kind="stable")[:n]
        selected = [records[int(x)] for x in order]
        row = summarize_slice(selected, allocator_table)
        row["retain_fraction"] = keep
        row["fit_margin_threshold"] = float(margins[order[-1]])
        row["fit_margin_mean"] = float(np.mean([r["fit_margin"] for r in selected]))
        out.append(row)
    return out


def main():
    rows = fetch_history()
    rounds = [r for r, _ in rows]
    draws = np.asarray([x for _, x in rows], dtype=int)

    rng = np.random.default_rng(SEED)
    sig_mean, sig_sd = calibrate_signature_scale(rng)
    records = build_capture_records(draws, rounds, sig_mean, sig_sd)
    allocator_table = build_allocator_table()

    out = {
        "probe": "LOTO6 Dynamics × Network Coupling v0",
        "source": URL,
        "rounds": [int(rounds[0]), int(rounds[-1])],
        "draws": len(rows),
        "historical_records": len(records),
        "question": "When the dynamics layer captures more of the realized six numbers inside a route-conditioned candidate pool, does a target-independent network allocator convert that capture into better 20-ticket outcomes than random arrangement, beyond the network gain expected for an equally compressed random pool?",
        "factorized_design": {
            "why": "Separate Dynamics material selection from Network arrangement. Historical strict-forward records provide only candidate capture K. A label-symmetric allocator response E[outcome | pool_size, K] is then measured independently, so target outcomes never tune the network.",
            "route_condition": "historical oracle correct-road, used only to isolate downstream coupling after route choice",
            "corridor_width": CORRIDOR_WIDTH,
            "fit_neighbors": FIT_NEIGHBORS,
            "pool_sizes": POOL_SIZES,
            "retention_curve": RETENTION,
            "allocator_seeds_per_pool_size_and_capture": ALLOCATOR_SEEDS,
            "tickets": TICKETS,
            "slots": TOTAL_SLOTS,
            "degree_rule": "same balanced 120-slot degree sequence is shared by Random allocator and Mesh allocator within each seed",
            "mesh_rule": "target-independent broad-mesh allocation minimizing repeated pairs while preserving exact number degrees; same allocator family as prior mesh probes",
        },
        "cells": {
            "global_random": "43-number material + random degree-preserving arrangement",
            "global_mesh": "43-number material + network mesh arrangement",
            "matched_random_random": "uniform random pool of Candidate's size + random arrangement",
            "matched_random_mesh": "same size-matched random material + network mesh arrangement",
            "candidate_random": "Dynamics route-conditioned Candidate material + random arrangement",
            "candidate_mesh": "same Candidate material + network mesh arrangement",
        },
        "primary_interaction": "(Candidate+Mesh - Candidate+Random) - (SizeMatchedRandom+Mesh - SizeMatchedRandom+Random). Positive means Network converts Dynamics-selected material better than equally compressed random material.",
        "allocator_response_table": allocator_table,
        "full_history": summarize_slice(records, allocator_table),
        "consistency_retention_curve": retention_curve(records, allocator_table),
        "signature_calibration": {"mean": sig_mean.tolist(), "sd": sig_sd.tolist()},
        "boundary": [
            "No Floot app changes were made by this probe.",
            "Correct-road identity is an oracle condition in historical evaluation. This isolates coupling after route choice and does not measure route-selection accuracy.",
            "Candidate construction is strict-forward: only earlier corridor examples form each pool. The target draw is used only to count how many of its six numbers were captured.",
            "Network allocation is measured in a separate canonical label-symmetric simulation conditioned only on pool size and capture K; it never sees historical target identities.",
            "The size-matched random-pool control is the main guard against mistaking compression alone for Dynamics × Network synergy.",
            "Historical LOTO6 has already been explored. Results are exploratory, not a pristine future holdout.",
            "If factorial interaction is near zero, the product should keep Dynamics and Network as separate modules even if each has standalone value.",
        ],
    }

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
