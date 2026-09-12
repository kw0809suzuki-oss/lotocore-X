from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

DATA = Path("data/loto6.csv")
STATE_OUT = Path("results/loto6_coarse_field_states.csv")
SUMMARY_OUT = Path("results/loto6_coarse_field_summary.csv")
BLOCK = 100
CENTER = 22.0
DRAW_SIZE = 6


def draw_center(row) -> float:
    return sum(float(row[f"n{i}"]) for i in range(1, DRAW_SIZE + 1)) / DRAW_SIZE


def position(z: float) -> str:
    if z > CENTER:
        return "U"
    if z < CENTER:
        return "D"
    return "C"


def movement(delta: float | None) -> str:
    if delta is None:
        return "NA"
    if delta > 0:
        return "UP"
    if delta < 0:
        return "DOWN"
    return "FLAT"


def ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def main() -> None:
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    full_blocks = len(df) // BLOCK
    if full_blocks < 4:
        raise ValueError(f"need at least {BLOCK * 4} draws, got {len(df)}")

    df = df.tail(full_blocks * BLOCK).reset_index(drop=True)
    df["center"] = df.apply(draw_center, axis=1)
    df["position"] = df["center"].map(position)
    df["delta"] = df["center"].diff()
    df["abs_delta"] = df["delta"].abs()
    df["movement"] = df["delta"].map(lambda x: movement(None if pd.isna(x) else float(x)))

    non_null_amp = df["abs_delta"].dropna()
    amp_threshold = float(non_null_amp.median())
    df["amplitude"] = df["abs_delta"].map(
        lambda x: "NA" if pd.isna(x) else ("ACTIVE" if float(x) > amp_threshold else "QUIET")
    )
    df["state"] = df["position"] + "|" + df["movement"] + "|" + df["amplitude"]

    summary_rows: list[dict] = []
    print("=== LOTO6 COARSE FIELD MAP: POSITION x MOVEMENT x AMPLITUDE ===")
    print(f"global amplitude split = median |delta center| = {amp_threshold:.4f}")

    for i in range(full_blocks):
        block = df.iloc[i * BLOCK : (i + 1) * BLOCK].copy().reset_index(drop=True)
        label = f"block{i + 1}"

        pos = Counter(x for x in block["position"] if x in {"U", "D"})
        mov = Counter(x for x in block["movement"] if x in {"UP", "DOWN", "FLAT"})
        amp = Counter(x for x in block["amplitude"] if x in {"QUIET", "ACTIVE"})
        state_counts = Counter(
            x for x in block["state"] if "NA" not in x and not x.startswith("C|")
        )

        pos_switch = pos_same = 0
        prev = None
        for cur in block["position"]:
            if cur not in {"U", "D"}:
                prev = None
                continue
            if prev is not None:
                if cur == prev:
                    pos_same += 1
                else:
                    pos_switch += 1
            prev = cur

        move_continue = move_reverse = 0
        prev = None
        for cur in block["movement"]:
            if cur not in {"UP", "DOWN"}:
                prev = None
                continue
            if prev is not None:
                if cur == prev:
                    move_continue += 1
                else:
                    move_reverse += 1
            prev = cur

        amp_same = amp_switch = 0
        prev = None
        for cur in block["amplitude"]:
            if cur not in {"QUIET", "ACTIVE"}:
                prev = None
                continue
            if prev is not None:
                if cur == prev:
                    amp_same += 1
                else:
                    amp_switch += 1
            prev = cur

        top_states = state_counts.most_common(3)
        top_text = "; ".join(f"{s}:{n}" for s, n in top_states)

        row = {
            "block": label,
            "round_start": int(block.iloc[0]["round"]),
            "round_end": int(block.iloc[-1]["round"]),
            "position_up_rate": ratio(pos["U"], pos["U"] + pos["D"]),
            "move_up_rate": ratio(mov["UP"], mov["UP"] + mov["DOWN"]),
            "active_rate": ratio(amp["ACTIVE"], amp["ACTIVE"] + amp["QUIET"]),
            "position_switch_rate": ratio(pos_switch, pos_switch + pos_same),
            "movement_reverse_rate": ratio(move_reverse, move_reverse + move_continue),
            "amplitude_switch_rate": ratio(amp_switch, amp_switch + amp_same),
            "top_states": top_text,
        }
        summary_rows.append(row)

        print(
            f"[{label}] rounds={row['round_start']}..{row['round_end']} "
            f"posU={row['position_up_rate']} moveUP={row['move_up_rate']} active={row['active_rate']} "
            f"posSwitch={row['position_switch_rate']} moveReverse={row['movement_reverse_rate']} "
            f"ampSwitch={row['amplitude_switch_rate']}"
        )
        print(f"  top states: {top_text}")

    STATE_OUT.parent.mkdir(parents=True, exist_ok=True)
    df[["round", "center", "position", "delta", "abs_delta", "movement", "amplitude", "state"]].to_csv(
        STATE_OUT, index=False
    )
    pd.DataFrame(summary_rows).to_csv(SUMMARY_OUT, index=False)

    print(f"saved -> {STATE_OUT}")
    print(f"saved -> {SUMMARY_OUT}")
    print("LOTO6_COARSE_FIELD_COMPLETE")


if __name__ == "__main__":
    main()
