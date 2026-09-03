from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

import lotocore
import x_agent

DATA = Path("data/loto7.csv")
OUT = Path("results/latest.csv")
NUMBERS = list(range(1, 38))


def actual_numbers(row) -> tuple[int, ...]:
    return tuple(sorted(int(row[f"n{i}"]) for i in range(1, 8)))


def metrics(pred: tuple[int, ...], actual: tuple[int, ...]) -> dict:
    hits = len(set(pred) & set(actual))
    pc = sum(pred) / 7.0
    ac = sum(actual) / 7.0
    pv = sum((x - pc) ** 2 for x in pred) / 7.0
    av = sum((x - ac) ** 2 for x in actual) / 7.0
    return {"hits": hits, "center_error": abs(pc-ac), "variance_error": abs(pv-av)}


def random_prediction(round_no: int) -> tuple[int, ...]:
    return tuple(sorted(random.Random(round_no).sample(NUMBERS, 7)))


def run(window: int = 100) -> pd.DataFrame:
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    rows = []
    for i in range(window, len(df)):
        history = df.iloc[i-window:i]
        actual = actual_numbers(df.iloc[i])
        rnd = int(df.iloc[i]["round"])
        predictions = {
            "lotocore": lotocore.predict(history),
            "x": x_agent.predict(history, competition_gate=True),
            "x_ungated": x_agent.predict(history, competition_gate=False),
            "x_spacing2": x_agent.predict(history, competition_gate=True, spacing_v2=True),
            "x_actionpoint": x_agent.predict(
                history, competition_gate=True, spacing_v2=True, action_point=True
            ),
            "random": None,
        }
        for model, pred_obj in predictions.items():
            if model == "random":
                pred, state = random_prediction(rnd), {"model":"random"}
            else:
                pred, state = pred_obj.numbers, pred_obj.state
            row = {"round":rnd,"date":df.iloc[i]["date"],"model":model,
                   "prediction":"-".join(map(str,pred)),"actual":"-".join(map(str,actual)),**metrics(pred,actual)}
            row.update({k:v for k,v in state.items() if k != "model"})
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> None:
    print("\n=== WALK FORWARD SUMMARY ===")
    for model in ("lotocore","x","x_ungated","x_spacing2","x_actionpoint","random"):
        g=results[results.model==model]
        print(f"{model:8s} n={len(g)} mean_hits={g.hits.mean():.4f} hit3+={(g.hits>=3).mean():.4f} best={g.hits.max()} mean_center_error={g.center_error.mean():.4f} mean_variance_error={g.variance_error.mean():.4f} hit_distribution={g.hits.value_counts().sort_index().to_dict()}")
    pivot=results.pivot(index="round",columns="model",values="hits")
    print(f"PAIRED X>LotoCore={(pivot.x>pivot.lotocore).sum()} X=LotoCore={(pivot.x==pivot.lotocore).sum()} X<LotoCore={(pivot.x<pivot.lotocore).sum()} | X>Random={(pivot.x>pivot.random).sum()} X=Random={(pivot.x==pivot.random).sum()} X<Random={(pivot.x<pivot.random).sum()}")
    print(f"GATE A/B Gate>Ungated={(pivot.x>pivot.x_ungated).sum()} Gate=Ungated={(pivot.x==pivot.x_ungated).sum()} Gate<Ungated={(pivot.x<pivot.x_ungated).sum()}")
    print(f"SPACING V2 VERSUS CHAMPION V2>Gate={(pivot.x_spacing2>pivot.x).sum()} V2=Gate={(pivot.x_spacing2==pivot.x).sum()} V2<Gate={(pivot.x_spacing2<pivot.x).sum()}")
    print(f"ACTION POINT VERSUS CHAMPION AP>Gate={(pivot.x_actionpoint>pivot.x).sum()} AP=Gate={(pivot.x_actionpoint==pivot.x).sum()} AP<Gate={(pivot.x_actionpoint<pivot.x).sum()}")
    ap=results[results.model=="x_actionpoint"]
    print(f"ACTION POINT FIRES={int(ap.spacing_engaged.sum())}/{len(ap)} rate={ap.spacing_engaged.mean():.4f} signal_mean={ap.action_signal.mean():.4f}")


def diagnose_x(results: pd.DataFrame) -> None:
    x=results[results.model=="x"].copy()
    print("\n=== X DIAGNOSTIC ===")
    # Does accurate Field geometry actually convert into number hits?
    x["center_band"]=pd.qcut(x.center_error,4,labels=["best","good","poor","worst"],duplicates="drop")
    for band,g in x.groupby("center_band",observed=True):
        print(f"CENTER {band}: n={len(g)} mean_hits={g.hits.mean():.3f} hit3+={(g.hits>=3).mean():.3f} center_err={g.center_error.mean():.3f} var_err={g.variance_error.mean():.2f}")
    # Which dynamic relation regime accompanies good/bad outcomes?
    x["vol_band"]=pd.qcut(x.volatility,3,labels=["low","mid","high"],duplicates="drop")
    for band,g in x.groupby("vol_band",observed=True):
        print(f"VOL {band}: n={len(g)} mean_hits={g.hits.mean():.3f} hit3+={(g.hits>=3).mean():.3f} center_err={g.center_error.mean():.3f} wp={g.w_persist.mean():.3f} wr={g.w_reverse.mean():.3f} wg={g.w_gap.mean():.3f}")
    # Observe Relation overlap and Weight Competition without changing predictions.
    # Shape and geometry both currently map to the outer/center axis; agreement can
    # therefore concentrate influence while disagreement can cancel it.
    x["shape_geom_mode"] = (
        ((x.shape_delta >= 0) == (x.gap_delta >= 0))
        .map({True: "aligned", False: "conflict"})
    )
    x["extra_weight"] = x.w_shape + x.w_geom
    x["weight_peak"] = x[["w_persist", "w_reverse", "w_gap", "w_shape", "w_geom"]].max(axis=1)
    print("WEIGHT COMPETITION:")
    for mode,g in x.groupby("shape_geom_mode"):
        print(
            f" {mode}: n={len(g)} mean_hits={g.hits.mean():.3f} "
            f"hit3+={(g.hits>=3).mean():.3f} center_err={g.center_error.mean():.3f} "
            f"var_err={g.variance_error.mean():.2f} "
            f"extra_w={g.extra_weight.mean():.3f} peak_w={g.weight_peak.mean():.3f}"
        )
    print(
        f" OVERLAP paired delta: aligned-conflict "
        f"hits={x[x.shape_geom_mode=='aligned'].hits.mean()-x[x.shape_geom_mode=='conflict'].hits.mean():+.3f} "
        f"var_err={x[x.shape_geom_mode=='aligned'].variance_error.mean()-x[x.shape_geom_mode=='conflict'].variance_error.mean():+.2f}"
    )
    print("TOP X rounds:")
    for _,r in x.sort_values(["hits","center_error"],ascending=[False,True]).head(8).iterrows():
        print(f" round={int(r['round'])} hits={int(r.hits)} center_err={r.center_error:.2f} var_err={r.variance_error:.2f} vol={r.volatility:.3f} W=({r.w_persist:.3f},{r.w_reverse:.3f},{r.w_gap:.3f}) pred={r.prediction} actual={r.actual}")
    print("LOW-CENTER / LOW-HIT residue:")
    residue=x[(x.center_error<=x.center_error.quantile(.25)) & (x.hits<=1)].sort_values("center_error").head(10)
    for _,r in residue.iterrows():
        print(f" round={int(r['round'])} hits={int(r.hits)} center_err={r.center_error:.2f} var_err={r.variance_error:.2f} vol={r.volatility:.3f} pred={r.prediction} actual={r.actual}")


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--window",type=int,default=100); args=p.parse_args()
    results=run(args.window); OUT.parent.mkdir(parents=True,exist_ok=True); results.to_csv(OUT,index=False)
    summarize(results); diagnose_x(results)
    print(f"saved -> {OUT}"); print("LOTOCORE_X_DIAGNOSTIC_COMPLETE")

if __name__=="__main__": main()
