from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

DATA = Path("data/loto6.csv")
OUT = Path("results/loto6_coarse_vs_random.csv")
DRAW_SIZE = 6
CENTER = 22.0
BLOCK = 100
SIMS = 1000
SEED = 20260912


def centers_from_df(df: pd.DataFrame) -> list[float]:
    return [
        sum(float(row[f"n{i}"]) for i in range(1, DRAW_SIZE + 1)) / DRAW_SIZE
        for _, row in df.iterrows()
    ]


def side(z: float) -> int:
    if z > CENTER:
        return 1
    if z < CENTER:
        return -1
    return 0


def sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def change_rate(seq: list[int]) -> float:
    pairs = [(a, b) for a, b in zip(seq, seq[1:]) if a != 0 and b != 0]
    if not pairs:
        return 0.0
    return sum(a != b for a, b in pairs) / len(pairs)


def block_metrics(centers: list[float], amp_threshold: float) -> dict[str, float]:
    pos = [side(z) for z in centers]
    deltas = [centers[i] - centers[i - 1] for i in range(1, len(centers))]
    moves = [sign(d) for d in deltas]
    amps = [1 if abs(d) >= amp_threshold else -1 for d in deltas]

    valid_pos = [x for x in pos if x != 0]
    valid_moves = [x for x in moves if x != 0]

    return {
        "posU": sum(x == 1 for x in valid_pos) / len(valid_pos) if valid_pos else 0.0,
        "moveUP": sum(x == 1 for x in valid_moves) / len(valid_moves) if valid_moves else 0.0,
        "active": sum(x == 1 for x in amps) / len(amps) if amps else 0.0,
        "posSwitch": change_rate(pos),
        "moveReverse": change_rate(moves),
        "ampSwitch": change_rate(amps),
    }


def summarize(centers: list[float], amp_threshold: float) -> tuple[dict[str, float], list[dict[str, float]]]:
    blocks = [centers[i:i + BLOCK] for i in range(0, BLOCK * 4, BLOCK)]
    bm = [block_metrics(b, amp_threshold) for b in blocks]
    names = list(bm[0])
    ranges = {name: max(x[name] for x in bm) - min(x[name] for x in bm) for name in names}

    full = block_metrics(centers, amp_threshold)
    ranges["overall_moveReverse"] = full["moveReverse"]
    ranges["overall_posSwitch"] = full["posSwitch"]
    return ranges, bm


def random_centers(rng: random.Random, n: int = 400) -> list[float]:
    nums = range(1, 44)
    out = []
    for _ in range(n):
        draw = rng.sample(nums, DRAW_SIZE)
        out.append(sum(draw) / DRAW_SIZE)
    return out


def main() -> None:
    df = pd.read_csv(DATA).sort_values("round").tail(400).reset_index(drop=True)
    if len(df) < 400:
        raise ValueError(f"need 400 draws, got {len(df)}")

    real_centers = centers_from_df(df)
    real_abs_delta = [abs(real_centers[i] - real_centers[i - 1]) for i in range(1, len(real_centers))]
    amp_threshold = float(pd.Series(real_abs_delta).median())
    real_summary, real_blocks = summarize(real_centers, amp_threshold)

    rng = random.Random(SEED)
    sim_rows = []
    for _ in range(SIMS):
        sim_summary, _ = summarize(random_centers(rng), amp_threshold)
        sim_rows.append(sim_summary)

    sim = pd.DataFrame(sim_rows)
    rows = []
    range_names = ["posU", "moveUP", "active", "posSwitch", "moveReverse", "ampSwitch"]
    for name in range_names:
        real_value = real_summary[name]
        tail = float((sim[name] >= real_value).mean())
        percentile = float((sim[name] <= real_value).mean())
        rows.append({
            "metric": f"block_range_{name}",
            "real": round(real_value, 4),
            "random_mean": round(float(sim[name].mean()), 4),
            "random_p95": round(float(sim[name].quantile(0.95)), 4),
            "tail_p_random_ge_real": round(tail, 4),
            "real_percentile": round(percentile, 4),
        })

    for name in ["overall_moveReverse", "overall_posSwitch"]:
        real_value = real_summary[name]
        tail = float((sim[name] >= real_value).mean())
        percentile = float((sim[name] <= real_value).mean())
        rows.append({
            "metric": name,
            "real": round(real_value, 4),
            "random_mean": round(float(sim[name].mean()), 4),
            "random_p95": round(float(sim[name].quantile(0.95)), 4),
            "tail_p_random_ge_real": round(tail, 4),
            "real_percentile": round(percentile, 4),
        })

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print("=== LOTO6 COARSE FIELD: REAL vs RANDOM ===")
    print(f"rounds={int(df.iloc[0]['round'])}..{int(df.iloc[-1]['round'])} sims={SIMS} seed={SEED}")
    print(f"fixed amplitude threshold from real data = {amp_threshold:.4f}")
    print("REAL 4 x 100 blocks:")
    for i, m in enumerate(real_blocks, 1):
        print(
            f"  block{i}: posU={m['posU']:.4f} moveUP={m['moveUP']:.4f} active={m['active']:.4f} "
            f"posSwitch={m['posSwitch']:.4f} moveReverse={m['moveReverse']:.4f} ampSwitch={m['ampSwitch']:.4f}"
        )
    print("COMPARISON (block-range; bigger means stronger block-to-block variation):")
    for r in rows:
        print(
            f"  {r['metric']}: real={r['real']} random_mean={r['random_mean']} "
            f"random_p95={r['random_p95']} tail_p={r['tail_p_random_ge_real']} percentile={r['real_percentile']}"
        )
    print(f"saved -> {OUT}")
    print("LOTO6_COARSE_VS_RANDOM_COMPLETE")


if __name__ == "__main__":
    main()
