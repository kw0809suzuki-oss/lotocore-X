from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

DATA = Path("data/loto6.csv")
OUT = Path("results/loto6_updown_transition.csv")
WINDOW = 100
CENTER = 22.0
DRAW_SIZE = 6


def draw_center(row) -> float:
    return sum(float(row[f"n{i}"]) for i in range(1, DRAW_SIZE + 1)) / DRAW_SIZE


def side(center: float) -> str | None:
    if center > CENTER:
        return "U"
    if center < CENTER:
        return "D"
    return None


def main() -> None:
    df = pd.read_csv(DATA).sort_values("round").tail(WINDOW).reset_index(drop=True)
    if len(df) < WINDOW:
        raise ValueError(f"need {WINDOW} draws, got {len(df)}")

    seq = []
    rows = []
    for _, row in df.iterrows():
        c = draw_center(row)
        s = side(c)
        seq.append(s)
        rows.append({"round": int(row["round"]), "center": round(c, 4), "side": s or "CENTER"})

    exact_center = sum(s is None for s in seq)
    side_counts = Counter(s for s in seq if s is not None)

    transition_next: dict[str, Counter] = defaultdict(Counter)
    valid_triples = 0
    next_counts = Counter()

    for i in range(2, len(seq)):
        a, b, c = seq[i - 2], seq[i - 1], seq[i]
        if a is None or b is None or c is None:
            continue
        key = f"{a}->{b}"
        transition_next[key][c] += 1
        next_counts[c] += 1
        valid_triples += 1

    baseline_up = next_counts["U"] / valid_triples if valid_triples else 0.0
    baseline_down = next_counts["D"] / valid_triples if valid_triples else 0.0

    summary_rows = []
    for key in ("D->D", "D->U", "U->D", "U->U"):
        counts = transition_next[key]
        n = counts["U"] + counts["D"]
        up_rate = counts["U"] / n if n else 0.0
        down_rate = counts["D"] / n if n else 0.0
        summary_rows.append(
            {
                "transition": key,
                "n": n,
                "next_up": counts["U"],
                "next_down": counts["D"],
                "next_up_rate": round(up_rate, 4),
                "next_down_rate": round(down_rate, 4),
                "up_lift_vs_baseline": round(up_rate - baseline_up, 4) if n else None,
            }
        )

    out = pd.DataFrame(summary_rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print("=== LOTO6 UP/DOWN TRANSITION CHECK ===")
    print(f"window={WINDOW} rounds={int(df.iloc[0]['round'])}..{int(df.iloc[-1]['round'])}")
    print(f"side_counts=U:{side_counts['U']} D:{side_counts['D']} exact_center_skipped:{exact_center}")
    print(f"valid_triples={valid_triples} baseline_next=U:{baseline_up:.4f} D:{baseline_down:.4f}")
    for r in summary_rows:
        print(
            f"{r['transition']} n={r['n']} nextU={r['next_up']} nextD={r['next_down']} "
            f"up_rate={r['next_up_rate']:.4f} lift={r['up_lift_vs_baseline']}"
        )
    print(f"saved -> {OUT}")
    print("LOTO6_UPDOWN_TRANSITION_COMPLETE")


if __name__ == "__main__":
    main()
