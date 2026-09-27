from __future__ import annotations

from collections import Counter
from pathlib import Path
import math

import pandas as pd

DATA = Path("data/loto7.csv")
OUT = Path("results/loto7_a_world_top7_one_swap_v0.csv")
NUMBERS = list(range(1, 38))
LONG_WINDOW = 100
RECENT_WINDOW = 20
ALPHA = 0.72
BOX_SIZE = 7
POOL_SIZE = 10


def draws_from_df(df):
    return [
        [int(row[f"n{i}"]) for i in range(1, 8)]
        for _, row in df.iterrows()
    ]


def frequency_distribution(draws):
    counts = Counter(n for draw in draws for n in draw)
    total = sum(counts.values())
    return {n: counts[n] / total for n in NUMBERS}


def mix_distribution(long_p, recent_p):
    raw = {
        n: ALPHA * long_p[n] + (1.0 - ALPHA) * recent_p[n]
        for n in NUMBERS
    }
    total = sum(raw.values())
    return {n: raw[n] / total for n in NUMBERS}


def ranked_numbers(p):
    return sorted(NUMBERS, key=lambda n: (-p[n], n))


def numeric_state(numbers):
    vals = list(numbers)
    center = sum(vals) / len(vals)
    variance = sum((n - center) ** 2 for n in vals) / len(vals)
    return center, math.sqrt(variance)


def distribution_state(p):
    center = sum(n * p[n] for n in NUMBERS)
    variance = sum(((n - center) ** 2) * p[n] for n in NUMBERS)
    return center, math.sqrt(variance)


def errors(box, a_center, a_spread):
    center, spread = numeric_state(box)
    return (
        abs(center - a_center),
        abs(spread - a_spread),
        center,
        spread,
    )


def mass(box, p):
    return sum(p[n] for n in box)


def main():
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    draws = draws_from_df(df)
    rows=[]

    for i in range(LONG_WINDOW, len(draws)):
        long_p = frequency_distribution(draws[i-LONG_WINDOW:i])
        recent_p = frequency_distribution(draws[i-RECENT_WINDOW:i])
        mix_p = mix_distribution(long_p, recent_p)
        ranked = ranked_numbers(mix_p)

        top7 = tuple(sorted(ranked[:BOX_SIZE]))
        pool10 = ranked[:POOL_SIZE]
        incoming = ranked[BOX_SIZE:POOL_SIZE]

        a_center, a_spread = distribution_state(mix_p)
        base_ce, base_se, base_c, base_s = errors(top7, a_center, a_spread)
        base_mass = mass(top7, mix_p)

        candidates=[]
        for out_num in ranked[:BOX_SIZE]:
            for in_num in incoming:
                cand = tuple(sorted((set(top7)-{out_num})|{in_num}))
                ce,se,c,s = errors(cand,a_center,a_spread)
                cm = mass(cand,mix_p)
                candidates.append({
                    "out":out_num, "in":in_num, "box":cand,
                    "center_error":ce, "spread_error":se,
                    "center":c, "spread":s,
                    "mass":cm,
                    "mass_loss":base_mass-cm,
                    "mass_loss_ratio":(base_mass-cm)/base_mass if base_mass else 0.0,
                    "center_improved":ce < base_ce - 1e-12,
                    "spread_improved":se < base_se - 1e-12,
                })

        both=[c for c in candidates if c["center_improved"] and c["spread_improved"]]
        any_center=[c for c in candidates if c["center_improved"]]
        any_spread=[c for c in candidates if c["spread_improved"]]

        def choose_best(items):
            if not items:
                return None
            return min(items, key=lambda c: (
                c["center_error"]/max(a_spread,1e-12) + c["spread_error"]/max(a_spread,1e-12),
                c["mass_loss"],
                c["out"], c["in"],
            ))

        best_both=choose_best(both)
        best_center=min(any_center,key=lambda c:(c["center_error"],c["mass_loss"])) if any_center else None
        best_spread=min(any_spread,key=lambda c:(c["spread_error"],c["mass_loss"])) if any_spread else None

        rec={
            "round":int(df.iloc[i]["round"]),
            "date":df.iloc[i].get("date",""),
            "top7":"-".join(f"{n:02d}" for n in top7),
            "rank8":ranked[7],
            "rank9":ranked[8],
            "rank10":ranked[9],
            "a_center":a_center,
            "a_spread":a_spread,
            "top7_center":base_c,
            "top7_spread":base_s,
            "top7_center_error":base_ce,
            "top7_spread_error":base_se,
            "top7_mass":base_mass,
            "both_improvement_exists":bool(both),
            "n_both_improving_swaps":len(both),
            "center_improvement_exists":bool(any_center),
            "spread_improvement_exists":bool(any_spread),
        }

        if best_both:
            rec.update({
                "best_out":best_both["out"],
                "best_in":best_both["in"],
                "best_box":"-".join(f"{n:02d}" for n in best_both["box"]),
                "best_center_error":best_both["center_error"],
                "best_spread_error":best_both["spread_error"],
                "best_mass":best_both["mass"],
                "best_mass_loss":best_both["mass_loss"],
                "best_mass_loss_ratio":best_both["mass_loss_ratio"],
                "center_error_reduction":base_ce-best_both["center_error"],
                "spread_error_reduction":base_se-best_both["spread_error"],
            })
        else:
            rec.update({
                "best_out":None,"best_in":None,"best_box":"",
                "best_center_error":None,"best_spread_error":None,
                "best_mass":None,"best_mass_loss":None,"best_mass_loss_ratio":None,
                "center_error_reduction":None,"spread_error_reduction":None,
            })

        if best_center:
            rec["best_center_only_mass_loss_ratio"]=best_center["mass_loss_ratio"]
            rec["best_center_only_error_reduction"]=base_ce-best_center["center_error"]
        else:
            rec["best_center_only_mass_loss_ratio"]=None
            rec["best_center_only_error_reduction"]=None

        if best_spread:
            rec["best_spread_only_mass_loss_ratio"]=best_spread["mass_loss_ratio"]
            rec["best_spread_only_error_reduction"]=base_se-best_spread["spread_error"]
        else:
            rec["best_spread_only_mass_loss_ratio"]=None
            rec["best_spread_only_error_reduction"]=None

        rows.append(rec)

    res=pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    res.to_csv(OUT,index=False)

    both=res[res["both_improvement_exists"]].copy()

    print("=== LOTO7 A-WORLD TOP7 ONE-SWAP v0 ===")
    print(f"rounds={len(res)} alpha={ALPHA:.2f} top7 neighborhood=rank8..10")
    print("No future target outcome is used.")
    print()
    print("EXISTENCE")
    print(f"both center+spread improvable={len(both)}/{len(res)} = {len(both)/len(res):.6f}")
    print(f"center improvable={int(res['center_improvement_exists'].sum())}/{len(res)} = {res['center_improvement_exists'].mean():.6f}")
    print(f"spread improvable={int(res['spread_improvement_exists'].sum())}/{len(res)} = {res['spread_improvement_exists'].mean():.6f}")
    print(f"mean n both-improving swaps={res['n_both_improving_swaps'].mean():.6f} / 21")
    print()
    print("WHEN BOTH CAN IMPROVE")
    if len(both):
        print(f"median mass loss ratio={both['best_mass_loss_ratio'].median():.6f}")
        print(f"mean mass loss ratio={both['best_mass_loss_ratio'].mean():.6f}")
        print(f"p90 mass loss ratio={both['best_mass_loss_ratio'].quantile(.90):.6f}")
        print(f"median center error reduction={both['center_error_reduction'].median():.6f}")
        print(f"median spread error reduction={both['spread_error_reduction'].median():.6f}")
        print(f"mean center error before={both['top7_center_error'].mean():.6f} after={both['best_center_error'].mean():.6f}")
        print(f"mean spread error before={both['top7_spread_error'].mean():.6f} after={both['best_spread_error'].mean():.6f}")
        print("incoming rank counts=" + str(both['best_in'].map(lambda n: {r:i+1 for i,r in enumerate(ranked_numbers(mix_distribution(frequency_distribution(draws[-LONG_WINDOW:]), frequency_distribution(draws[-RECENT_WINDOW:]))))}.get(n,0)).value_counts().sort_index().to_dict()) if False else "recorded per round")
    print()
    print("MASS LOSS BANDS FOR BOTH-IMPROVING BEST SWAP")
    if len(both):
        for pct in (0.0025,0.005,0.01,0.02,0.05):
            print(f"<= {pct*100:.2f}% loss: {(both['best_mass_loss_ratio']<=pct).mean():.6f}")
    print()
    print("RECENT 15")
    cols=["round","top7","both_improvement_exists","n_both_improving_swaps","best_out","best_in","best_box","best_mass_loss_ratio","center_error_reduction","spread_error_reduction"]
    print(res[cols].tail(15).to_string(index=False))
    print(f"saved -> {OUT}")


if __name__=="__main__":
    main()
