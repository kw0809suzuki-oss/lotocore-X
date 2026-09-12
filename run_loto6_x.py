from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import loto6_x_agent

DATA = Path("data/loto6.csv")
OUT = Path("results/loto6_x_tickets.csv")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=15)
    args = p.parse_args()

    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    tickets, state = loto6_x_agent.generate_tickets(df, count=args.count)

    print("=== LOTO6 DYNAMIC X ===")
    print(f"latest_round={int(df.iloc[-1]['round'])}")
    for key in (
        "field_regime",
        "field_band",
        "field_bands",
        "field_mean_abs_move",
        "field_reversals",
        "field_stay_high",
        "field_stay_low",
        "analog_count",
        "w_analog",
        "w_persist",
        "w_reverse",
        "w_gap",
        "ranked_top18",
    ):
        print(f"{key}={state.get(key)}")

    rows = []
    for i, ticket in enumerate(tickets, start=1):
        text = "-".join(f"{n:02d}" for n in ticket)
        print(f"{i:02d}: {text}")
        rows.append(
            {
                "ticket": i,
                "numbers": text,
                "latest_round": int(df.iloc[-1]["round"]),
                **state,
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"saved -> {OUT}")
    print("LOTO6_X_COMPLETE")


if __name__ == "__main__":
    main()
