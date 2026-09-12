from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

DATA = Path("data/loto6.csv")
OUT = Path("results/loto6_updown_transition.csv")
BLOCK = 100
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


def analyze_block(df: pd.DataFrame, label: str) -> tuple[list[dict], dict]:
    df = df.reset_index(drop=True)
    seq = [side(draw_center(row)) for _, row in df.iterrows()]

    exact_center = sum(s is None for s in seq)
    side_counts = Counter(s for s in seq if s is not None)
    transition_next: dict[str, Counter] = defaultdict(Counter)
    next_counts = Counter()
    valid_triples = 0

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

    rows = []
    for key in ("D->D", "D->U", "U->D", "U->U"):
        counts = transition_next[key]
        n = counts["U"] + counts["D"]
        up_rate = counts["U"] / n if n else 0.0
        down_rate = counts["D"] / n if n else 0.0
        rows.append(
            {
                "block": label,
                "round_start": int(df.iloc[0]["round"]),
                "round_end": int(df.iloc[-1]["round"]),
                "transition": key,
                "n": n,
                "next_up": counts["U"],
                "next_down": counts["D"],
                "next_up_rate": round(up_rate, 4),
                "next_down_rate": round(down_rate, 4),
                "up_lift_vs_baseline": round(up_rate - baseline_up, 4) if n else None,
            }
        )

    meta = {
        "label": label,
        "round_start": int(df.iloc[0]["round"]),
        "round_end": int(df.iloc[-1]["round"]),
        "U": side_counts["U"],
        "D": side_counts["D"],
        "exact_center": exact_center,
        "valid_triples": valid_triples,
        "baseline_up": baseline_up,
        "baseline_down": baseline_down,
    }
    return rows, meta


def main() -> None:
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    full_blocks = len(df) // BLOCK
    if full_blocks < 2:
        raise ValueError(f"need at least {BLOCK * 2} draws, got {len(df)}")

    df = df.tail(full_blocks * BLOCK).reset_index(drop=True)
    blocks = []
    for i in range(full_blocks):
        block_df = df.iloc[i * BLOCK : (i + 1) * BLOCK]
        blocks.append((block_df, f"block{i + 1}"))

    all_rows = []
    metas = []
    for block_df, label in blocks:
        rows, meta = analyze_block(block_df, label)
        all_rows.extend(rows)
        metas.append(meta)

    out = pd.DataFrame(all_rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"=== LOTO6 UP/DOWN TRANSITION CHECK: {full_blocks} x 100-DRAW BLOCKS ===")
    total_du_n = 0
    total_du_up = 0
    total_du_down = 0
    for meta in metas:
        print(
            f"[{meta['label']}] rounds={meta['round_start']}..{meta['round_end']} "
            f"U:{meta['U']} D:{meta['D']} exact_center_skipped:{meta['exact_center']} "
            f"valid_triples={meta['valid_triples']} baselineU={meta['baseline_up']:.4f}"
        )
        block_rows = [x for x in all_rows if x["block"] == meta["label"]]
        for r in block_rows:
            print(
                f"  {r['transition']} n={r['n']} nextU={r['next_up']} nextD={r['next_down']} "
                f"up_rate={r['next_up_rate']:.4f} lift={r['up_lift_vs_baseline']}"
            )
            if r["transition"] == "D->U":
                total_du_n += r["n"]
                total_du_up += r["next_up"]
                total_du_down += r["next_down"]

    pooled_rate = total_du_up / total_du_n if total_du_n else 0.0
    print(
        f"POOLED D->U: n={total_du_n} nextU={total_du_up} nextD={total_du_down} "
        f"up_rate={pooled_rate:.4f}"
    )
    print(f"saved -> {OUT}")
    print("LOTO6_UPDOWN_TRANSITION_COMPLETE")


if __name__ == "__main__":
    main()
