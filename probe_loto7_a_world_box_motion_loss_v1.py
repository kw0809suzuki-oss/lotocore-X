from __future__ import annotations

from pathlib import Path
import pandas as pd

from probe_loto7_a_world_box_motion_v0 import (
    DATA, NUMBERS, LONG_WINDOW, RECENT_WINDOW, ALPHA, BOX_SIZE, AXES,
    draws_from_df, frequency_distribution, mix_distribution,
    box_from_distribution, box_distribution, state, sign,
)

OUT = Path("results/loto7_a_world_box_motion_loss_v1.csv")


def main():
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    draws = draws_from_df(df)

    states = []
    for i in range(LONG_WINDOW, len(draws)):
        long_p = frequency_distribution(draws[i - LONG_WINDOW:i])
        recent_p = frequency_distribution(draws[i - RECENT_WINDOW:i])
        mix_p = mix_distribution(long_p, recent_p)
        box = box_from_distribution(mix_p)
        box_p = box_distribution(box)
        a = state(mix_p)
        b = state(box_p)
        states.append({
            "round": int(df.iloc[i]["round"]),
            "box_tuple": box,
            **{f"a_{k}": v for k, v in a.items()},
            **{f"box_{k}": v for k, v in b.items()},
        })

    s = pd.DataFrame(states)
    rows = []
    for j in range(1, len(s)):
        prev = s.iloc[j - 1]
        cur = s.iloc[j]
        prev_box = set(prev["box_tuple"])
        cur_box = set(cur["box_tuple"])
        replacements = len(prev_box - cur_box)
        same_box = replacements == 0
        rec = {
            "from_round": int(prev["round"]),
            "to_round": int(cur["round"]),
            "same_box": same_box,
            "replacements": replacements,
        }
        for axis in AXES:
            da = float(cur[f"a_{axis}"] - prev[f"a_{axis}"])
            db = float(cur[f"box_{axis}"] - prev[f"box_{axis}"])
            sa = sign(da)
            sb = sign(db)
            active = sa != 0
            match = sa == sb
            rec[f"a_d_{axis}"] = da
            rec[f"box_d_{axis}"] = db
            rec[f"a_active_{axis}"] = active
            rec[f"match_{axis}"] = match
            rec[f"box_flat_{axis}"] = sb == 0
            rec[f"mismatch_due_box_flat_{axis}"] = active and sb == 0
            rec[f"mismatch_after_box_move_{axis}"] = active and sb != 0 and sa != sb
        rows.append(rec)

    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)

    print("=== LOTO7 A-WORLD BOX MOTION LOSS v1 ===")
    print(f"transitions={len(res)} alpha={ALPHA:.2f} box={BOX_SIZE}")
    print("No future target outcome is used.")
    print()

    print("BOX CHANGE")
    same = res["same_box"]
    print(f"same box={int(same.sum())}/{len(res)} = {same.mean():.6f}")
    print(f"changed box={int((~same).sum())}/{len(res)} = {(~same).mean():.6f}")
    print("replacements_dist=" + str(res["replacements"].value_counts().sort_index().to_dict()))

    print()
    print("MISMATCH DECOMPOSITION")
    for axis in AXES:
        active = res[f"a_active_{axis}"]
        n_active = int(active.sum())
        mism = active & (~res[f"match_{axis}"])
        flat_loss = res[f"mismatch_due_box_flat_{axis}"]
        moved_wrong = res[f"mismatch_after_box_move_{axis}"]
        print(
            f"{axis}: active={n_active} mismatches={int(mism.sum())} "
            f"box-flat-loss={int(flat_loss.sum())} "
            f"changed-but-wrong={int(moved_wrong.sum())}"
        )

    print()
    print("DIRECTION AGREEMENT CONDITIONAL ON BOX CHANGE")
    changed = ~res["same_box"]
    for axis in AXES:
        part = res[changed & res[f"a_active_{axis}"]]
        match = part[f"match_{axis}"].mean() if len(part) else float("nan")
        pearson = part[f"a_d_{axis}"].corr(part[f"box_d_{axis}"]) if len(part) > 1 else float("nan")
        print(f"{axis}: n={len(part)} match={match:.6f} pearson={pearson:.6f}")

    print()
    print("A-WORLD MOTION MAGNITUDE")
    for axis in AXES:
        same_mag = res.loc[same, f"a_d_{axis}"].abs().mean()
        change_mag = res.loc[~same, f"a_d_{axis}"].abs().mean()
        print(f"{axis}: |delta| same-box={same_mag:.6f} changed-box={change_mag:.6f}")

    print()
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
