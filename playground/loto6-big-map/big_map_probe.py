from __future__ import annotations

import csv
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import requests

URL = "https://www.mk-mode.com/rails/loto/LOTO6_ALL.csv"
OUT = Path("results/loto6_big_map_v0.json")

SEED = 20260924
BURN = 100
CENTER = 37 / 6
CENTER_BAND = 0.25
NULL_WORLDS = 120
CAL_DRAWS = 200000

PHASES = ["Flow", "Break", "Uncertainty", "Re-formation"]
MU_BINS = ["L", "M", "H"]
VAR_BINS = ["C", "N", "W"]  # compact / normal / wide
NODES = [f"{p}|{m}{v}" for p in PHASES for m in MU_BINS for v in VAR_BINS]
NODE_INDEX = {n:i for i,n in enumerate(NODES)}


def fetch_history():
    r = requests.get(URL, timeout=30, headers={"User-Agent": "lotocore-X big-map probe"})
    r.raise_for_status()
    try:
        text = r.content.decode("cp932")
    except UnicodeDecodeError:
        text = r.content.decode("utf-8")

    rows = []
    reader = csv.reader(io.StringIO(text))
    next(reader, None)
    for row in reader:
        try:
            rnd = int(row[0])
            nums = [int(row[2 + i]) for i in range(6)]
        except Exception:
            continue
        if len(nums) == 6 and len(set(nums)) == 6 and all(1 <= n <= 43 for n in nums):
            rows.append((rnd, sorted(nums)))
    rows.sort(key=lambda x: x[0])
    if len(rows) < 1000:
        raise RuntimeError(f"too few parsed draws: {len(rows)}")
    return rows


def random_world(rng, n):
    u = rng.random((n, 43))
    idx = np.argpartition(u, 6, axis=1)[:, :6] + 1
    return np.sort(idx, axis=1)


def shape(draw):
    a = np.asarray(draw, dtype=float)
    return float(a.mean()), float(a.var())


def calibrate_shape(rng):
    x = random_world(rng, CAL_DRAWS)
    mus = x.mean(axis=1)
    vars_ = x.var(axis=1)
    qmu = np.quantile(mus, [1/3, 2/3]).tolist()
    qvar = np.quantile(vars_, [1/3, 2/3]).tolist()
    return {
        "mu_q33": float(qmu[0]),
        "mu_q67": float(qmu[1]),
        "var_q33": float(qvar[0]),
        "var_q67": float(qvar[1]),
        "mu_mean": float(mus.mean()),
        "mu_sd": float(mus.std(ddof=1)),
        "var_mean": float(vars_.mean()),
        "var_sd": float(vars_.std(ddof=1)),
    }


def bin3(x, q1, q2, labels):
    if x < q1:
        return labels[0]
    if x <= q2:
        return labels[1]
    return labels[2]


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


def entropy_from_counts(counts):
    arr = np.asarray(list(counts.values()), dtype=float)
    if arr.sum() <= 0:
        return 0.0
    p = arr / arr.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def mean_run_lengths(labels):
    by = defaultdict(list)
    if not labels:
        return {}
    cur = labels[0]
    n = 1
    for x in labels[1:]:
        if x == cur:
            n += 1
        else:
            by[cur].append(n)
            cur, n = x, 1
    by[cur].append(n)
    return {k: float(np.mean(v)) for k, v in by.items()}


def evaluate(draws, cal, rounds=None):
    draws = np.asarray(draws, dtype=int)
    shapes = [shape(d) for d in draws]

    ages = np.zeros(43, dtype=float)
    prev_A = None
    prev_label = "Uncertainty"

    phase_labels = []
    shape_labels = []
    nodes = []
    node_counts = Counter()
    edge_counts = Counter()
    phase_edges = Counter()
    phase_shape_step = defaultdict(list)
    rows = []
    prev_node = None
    prev_phase_eval = None
    prev_shape_z = None

    for i, draw in enumerate(draws):
        A = float(ages.mean())
        phase = classify_phase(A, prev_A, prev_label)

        # pre-draw shape state = last observed draw's shape
        if i == 0:
            mu, vv = cal["mu_mean"], cal["var_mean"]
        else:
            mu, vv = shapes[i - 1]

        mb = bin3(mu, cal["mu_q33"], cal["mu_q67"], MU_BINS)
        vb = bin3(vv, cal["var_q33"], cal["var_q67"], VAR_BINS)
        shape_label = f"{mb}{vb}"
        node = f"{phase}|{shape_label}"

        z = np.array([
            (mu - cal["mu_mean"]) / max(cal["mu_sd"], 1e-9),
            (vv - cal["var_mean"]) / max(cal["var_sd"], 1e-9),
        ], dtype=float)

        if i >= BURN:
            phase_labels.append(phase)
            shape_labels.append(shape_label)
            nodes.append(node)
            node_counts[node] += 1

            if prev_node is not None:
                edge_counts[(prev_node, node)] += 1
            if prev_phase_eval is not None:
                phase_edges[(prev_phase_eval, phase)] += 1
            if prev_shape_z is not None:
                phase_shape_step[phase].append(float(np.linalg.norm(z - prev_shape_z)))

            rows.append({
                "round": int(rounds[i]) if rounds is not None else i + 1,
                "phase": phase,
                "shape": shape_label,
                "node": node,
                "A": A,
                "mu_prev": mu,
                "var_prev": vv,
            })
            prev_node = node
            prev_phase_eval = phase
            prev_shape_z = z

        next_ages = ages + 1
        for n in draw:
            next_ages[int(n) - 1] = 0
        ages = next_ages
        prev_A = A
        prev_label = phase

    n = len(nodes)
    node_freq = {k: node_counts[k] / n for k in NODES}
    phase_occ = {p: phase_labels.count(p) / n for p in PHASES}
    shape_states = [f"{m}{v}" for m in MU_BINS for v in VAR_BINS]
    shape_occ = {s: shape_labels.count(s) / n for s in shape_states}

    edge_total = max(1, len(nodes) - 1)
    edge_freq = {f"{a}->{b}": c / edge_total for (a,b), c in edge_counts.items()}
    phase_edge_total = max(1, len(phase_labels) - 1)
    phase_edge_freq = {f"{a}->{b}": c / phase_edge_total for (a,b), c in phase_edges.items()}

    phase_shape_step_mean = {
        p: float(np.mean(phase_shape_step[p])) if phase_shape_step[p] else None for p in PHASES
    }

    return {
        "n": n,
        "phase_occupancy": phase_occ,
        "shape_occupancy": shape_occ,
        "node_frequency": node_freq,
        "edge_frequency_sparse": edge_freq,
        "phase_edge_frequency": phase_edge_freq,
        "phase_run_mean": mean_run_lengths(phase_labels),
        "shape_run_mean": mean_run_lengths(shape_labels),
        "node_entropy_bits": entropy_from_counts(node_counts),
        "phase_entropy_bits": entropy_from_counts(Counter(phase_labels)),
        "shape_entropy_bits": entropy_from_counts(Counter(shape_labels)),
        "phase_shape_step_mean": phase_shape_step_mean,
        "rows": rows,
    }


def summarize_null(actual, worlds):
    out = {}

    # Scalar summaries
    for field in ["node_entropy_bits", "phase_entropy_bits", "shape_entropy_bits"]:
        vals = np.asarray([w[field] for w in worlds], dtype=float)
        a = actual[field]
        out[field] = {
            "actual": a,
            "null_mean": float(vals.mean()),
            "null_q025": float(np.quantile(vals, .025)),
            "null_q975": float(np.quantile(vals, .975)),
            "percentile": float(np.mean(vals <= a)),
        }

    # Phase occupancy
    out["phase_occupancy"] = {}
    for p in PHASES:
        vals = np.asarray([w["phase_occupancy"][p] for w in worlds], dtype=float)
        a = actual["phase_occupancy"][p]
        out["phase_occupancy"][p] = {
            "actual": a,
            "null_mean": float(vals.mean()),
            "null_q025": float(np.quantile(vals, .025)),
            "null_q975": float(np.quantile(vals, .975)),
            "percentile": float(np.mean(vals <= a)),
        }

    # Mean shape step by phase
    out["phase_shape_step_mean"] = {}
    for p in PHASES:
        vals = np.asarray([w["phase_shape_step_mean"][p] for w in worlds if w["phase_shape_step_mean"][p] is not None], dtype=float)
        a = actual["phase_shape_step_mean"][p]
        out["phase_shape_step_mean"][p] = {
            "actual": a,
            "null_mean": float(vals.mean()),
            "null_q025": float(np.quantile(vals, .025)),
            "null_q975": float(np.quantile(vals, .975)),
            "percentile": float(np.mean(vals <= a)),
        }

    # Composite node map: compare each of 36 frequencies to B cloud.
    node_rows = []
    for node in NODES:
        vals = np.asarray([w["node_frequency"][node] for w in worlds], dtype=float)
        a = actual["node_frequency"][node]
        sd = float(vals.std(ddof=1))
        z = (a - float(vals.mean())) / sd if sd > 1e-12 else 0.0
        node_rows.append({
            "node": node,
            "actual": a,
            "null_mean": float(vals.mean()),
            "delta": a - float(vals.mean()),
            "z": float(z),
            "percentile": float(np.mean(vals <= a)),
        })
    node_rows.sort(key=lambda x: abs(x["z"]), reverse=True)
    out["top_node_residuals"] = node_rows[:12]

    # Composite transition map: evaluate all observed actual edges with null expectation >= ~1 event.
    all_edges = set()
    for w in worlds:
        all_edges.update(w["edge_frequency_sparse"].keys())
    all_edges.update(actual["edge_frequency_sparse"].keys())
    edge_rows = []
    for edge in all_edges:
        vals = np.asarray([w["edge_frequency_sparse"].get(edge, 0.0) for w in worlds], dtype=float)
        a = actual["edge_frequency_sparse"].get(edge, 0.0)
        mean = float(vals.mean())
        sd = float(vals.std(ddof=1))
        expected_count = mean * max(1, actual["n"] - 1)
        if expected_count < 1.0 and a == 0:
            continue
        z = (a - mean) / sd if sd > 1e-12 else 0.0
        edge_rows.append({
            "edge": edge,
            "actual": a,
            "null_mean": mean,
            "expected_count": expected_count,
            "delta": a - mean,
            "z": float(z),
            "percentile": float(np.mean(vals <= a)),
        })
    edge_rows.sort(key=lambda x: abs(x["z"]), reverse=True)
    out["top_edge_residuals"] = edge_rows[:15]

    # Macro phase transitions
    phase_edges = [f"{a}->{b}" for a in PHASES for b in PHASES]
    pe_rows = []
    for edge in phase_edges:
        vals = np.asarray([w["phase_edge_frequency"].get(edge,0.0) for w in worlds], dtype=float)
        a = actual["phase_edge_frequency"].get(edge,0.0)
        mean = float(vals.mean())
        sd = float(vals.std(ddof=1))
        z = (a - mean) / sd if sd > 1e-12 else 0.0
        pe_rows.append({
            "edge": edge, "actual": a, "null_mean": mean, "delta": a-mean,
            "z": float(z), "percentile": float(np.mean(vals <= a))
        })
    pe_rows.sort(key=lambda x: abs(x["z"]), reverse=True)
    out["phase_transition_residuals"] = pe_rows

    return out


def main():
    rows = fetch_history()
    rounds = [r for r,_ in rows]
    actual_draws = np.asarray([x for _,x in rows], dtype=int)

    rng = np.random.default_rng(SEED)
    cal = calibrate_shape(rng)
    actual = evaluate(actual_draws, cal, rounds)

    worlds = []
    for _ in range(NULL_WORLDS):
        worlds.append(evaluate(random_world(rng, len(rows)), cal))

    comparison = summarize_null(actual, worlds)

    compact_actual = dict(actual)
    compact_actual.pop("rows")

    out = {
        "probe": "LOTO6 Big Map v0",
        "source": URL,
        "rounds": [rounds[0], rounds[-1]],
        "draws": len(rows),
        "burn": BURN,
        "map_definition": {
            "time_axis": "Flow / Break / Uncertainty / Re-formation from mean-age movement, same coarse classifier as prior State Map",
            "shape_axis": "pre-draw previous-result shape = centroid x variance, each axis split into B-world terciles",
            "shape_codes": {
                "L/M/H": "centroid low / middle / high relative to B terciles",
                "C/N/W": "variance compact / normal / wide relative to B terciles",
            },
            "composite_nodes": "4 phase states x 9 shape states = 36-node map",
            "transition": "one pre-draw composite state to the next",
        },
        "B_shape_calibration": cal,
        "actual": compact_actual,
        "null": {
            "worlds": NULL_WORLDS,
            "seed": SEED,
            "generator": "independent exact-uniform 6-of-43 worlds, identical state map and shape calibration",
        },
        "comparison": comparison,
        "boundary": [
            "Large-map observation only; no ticket selector is being tuned here.",
            "Node/edge residuals are descriptive candidates, not predictive evidence.",
            "Historical A is exploratory and includes research-contaminated periods.",
            "The purpose is to see whether A occupies or traverses the same large-scale map as B, and where any coherent residual geometry remains.",
        ],
    }

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
