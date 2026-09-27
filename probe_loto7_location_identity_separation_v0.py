from __future__ import annotations

from itertools import product
from math import floor, ceil
from pathlib import Path

import pandas as pd

import lotocore

DATA = Path("data/loto7.csv")
OUT = Path("results/loto7_location_identity_separation_v0.csv")
SUMMARY = Path("results/loto7_location_identity_separation_summary_v0.csv")
WINDOW = 100


def actual(row):
    return {int(row[f"n{i}"]) for i in range(1, 8)}


def nearest_distance(truth, points):
    vals = list(points)
    return sum(min(abs(a - x) for x in vals) for a in truth) / 7.0


def floating_positions(history):
    base = tuple(sorted(lotocore.predict(history).numbers))
    snap = lotocore.score_snapshot(history)
    scores = {int(k): float(v) for k, v in snap["scores"].items()}

    positions = []
    freedoms = []
    residuals = []
    for n in base:
        local = [k for k in (n - 1, n, n + 1) if 1 <= k <= 37]
        denom = sum(scores[k] for k in local)
        p = n if denom <= 0 else sum(k * scores[k] for k in local) / denom
        r = p - n
        positions.append(p)
        residuals.append(r)

        if abs(r) < 1e-12:
            freedoms.append((n,))
        else:
            adj = n + (1 if r > 0 else -1)
            adj = min(37, max(1, adj))
            freedoms.append(tuple(sorted(set((n, adj)))))

    return base, positions, residuals, freedoms, scores


def decode_joint(positions, freedoms, scores):
    best = None
    for cand in product(*freedoms):
        if len(set(cand)) < 7:
            continue
        ordered = tuple(sorted(cand))

        # Keep the seven-number total near the floating-state total first.
        total_error = abs(sum(ordered) - sum(positions))
        local_error = sum((x - p) ** 2 for x, p in zip(cand, positions))
        score_mass = sum(scores[x] for x in cand)
        key = (total_error, local_error, -score_mass, ordered)

        if best is None or key < best[0]:
            best = (key, ordered)

    if best is None:
        # Rare collision fallback: nearest distinct integers globally.
        used = set()
        out = []
        for p in positions:
            candidates = sorted(range(1, 38), key=lambda x: (abs(x - p), -scores[x], x))
            x = next(v for v in candidates if v not in used)
            used.add(x)
            out.append(x)
        return tuple(sorted(out))

    return best[1]


def main():
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    rows = []

    for i in range(WINDOW, len(df)):
        history = df.iloc[i - WINDOW:i]
        truth = actual(df.iloc[i])

        base, floating, residuals, freedoms, scores = floating_positions(history)
        decoded = decode_joint(floating, freedoms, scores)

        base_d = nearest_distance(truth, base)
        float_d = nearest_distance(truth, floating)
        dec_d = nearest_distance(truth, decoded)

        rows.append({
            "target_round": int(df.iloc[i]["round"]),
            "base_hits": len(set(base) & truth),
            "decoded_hits": len(set(decoded) & truth),
            "base_distance": base_d,
            "floating_distance": float_d,
            "decoded_distance": dec_d,
            "location_gain_vs_base": base_d - float_d,
            "integerization_loss": dec_d - float_d,
            "decoded_gain_vs_base": base_d - dec_d,
            "mean_abs_residual": sum(abs(r) for r in residuals) / 7.0,
            "base": "-".join(f"{n:02d}" for n in base),
            "floating": "|".join(f"{p:.4f}" for p in floating),
            "residuals": "|".join(f"{r:+.4f}" for r in residuals),
            "freedom": "|".join("/".join(f"{n:02d}" for n in f) for f in freedoms),
            "decoded": "-".join(f"{n:02d}" for n in decoded),
            "actual": "-".join(f"{n:02d}" for n in sorted(truth)),
        })

    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)

    summary = pd.DataFrame([{
        "targets": len(res),
        "base_mean_hits": res.base_hits.mean(),
        "decoded_mean_hits": res.decoded_hits.mean(),
        "base_mean_distance": res.base_distance.mean(),
        "floating_mean_distance": res.floating_distance.mean(),
        "decoded_mean_distance": res.decoded_distance.mean(),
        "mean_location_gain_vs_base": res.location_gain_vs_base.mean(),
        "mean_integerization_loss": res.integerization_loss.mean(),
        "mean_decoded_gain_vs_base": res.decoded_gain_vs_base.mean(),
        "mean_abs_residual": res.mean_abs_residual.mean(),
        "floating_better_than_base_rate": (res.floating_distance < res.base_distance).mean(),
        "decoded_better_than_base_rate": (res.decoded_distance < res.base_distance).mean(),
        "decoded_hit_improve_rate": (res.decoded_hits > res.base_hits).mean(),
        "decoded_hit_worse_rate": (res.decoded_hits < res.base_hits).mean(),
    }])
    summary.to_csv(SUMMARY, index=False)

    print("=== LOTO7 LOCATION-IDENTITY SEPARATION v0 ===")
    print(f"targets={len(res)} rounds={int(res.target_round.min())}..{int(res.target_round.max())}")
    print("CORE is frozen. Only ±1 local score centroid + joint integer decode is added.")
    print()
    print(f"mean hits: base={res.base_hits.mean():.6f} decoded={res.decoded_hits.mean():.6f} delta={(res.decoded_hits-res.base_hits).mean():+.6f}")
    print(f"mean nearest distance: base={res.base_distance.mean():.6f}")
    print(f"                       floating={res.floating_distance.mean():.6f}")
    print(f"                       decoded={res.decoded_distance.mean():.6f}")
    print(f"location gain vs base={res.location_gain_vs_base.mean():+.6f}")
    print(f"integerization loss={res.integerization_loss.mean():+.6f}")
    print(f"decoded gain vs base={res.decoded_gain_vs_base.mean():+.6f}")
    print(f"mean abs residual={res.mean_abs_residual.mean():.6f}")
    print(f"floating better than base={(res.floating_distance < res.base_distance).mean():.6f}")
    print(f"decoded better than base={(res.decoded_distance < res.base_distance).mean():.6f}")
    print()
    print("RECENT 20")
    print(res.tail(20).to_string(index=False))
    print(f"saved -> {OUT}")
    print(f"saved -> {SUMMARY}")


if __name__ == "__main__":
    main()
