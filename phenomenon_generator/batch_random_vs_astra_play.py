from __future__ import annotations

import json
import pandas as pd

from compare_random_vs_astra_play import run


SEEDS = [20261005, 20261006, 20261007, 20261008, 20261009, 20261010]


def main() -> None:
    df = pd.read_csv("data/loto7.csv")
    rows = []
    for seed in SEEDS:
        out = run(df, last_n=100, seed=seed)
        diff = out["difference_astra_minus_random"]
        rows.append({
            "seed": seed,
            **diff,
        })

    metrics = [
        "mean_max_hit",
        "max_hit_4plus_rounds",
        "ticket_3plus_total",
        "ticket_4plus_total",
        "ticket_5plus_total",
    ]
    summary = {}
    for m in metrics:
        vals = [r[m] for r in rows]
        summary[m] = {
            "mean_diff": sum(vals) / len(vals),
            "astra_positive_seeds": sum(v > 0 for v in vals),
            "equal_seeds": sum(v == 0 for v in vals),
            "random_positive_seeds": sum(v < 0 for v in vals),
            "diffs": vals,
        }

    print(json.dumps({
        "name": "Random20 vs Astra20 multi-seed play",
        "seeds": SEEDS,
        "per_seed_differences": rows,
        "summary": summary,
        "boundary": "Play comparison only; same 100-round window and same Astra rule, varying seed only."
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
