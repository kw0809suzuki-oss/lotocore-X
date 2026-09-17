"""LOTO7 residual play: find transitions that fall outside prior CORE-like tendency.

Observation only. Uses only history available before each target draw.
State = (centroid, variance), matching the coarse LOTO CORE observation axes.
A residual is NOT a prediction failure; it is a realized next State unusually far
from what similar prior States tended to do.
"""
from __future__ import annotations

import math
import statistics
from fetch_loto7 import fetch

MIN_HISTORY = 60
K = 15
ROBUST_Z = 3.0


def state(nums):
    mu = sum(nums) / 7.0
    var = sum((x - mu) ** 2 for x in nums) / 7.0
    return (mu, var)


def med(xs):
    return statistics.median(xs)


def mad(xs):
    m = med(xs)
    return med([abs(x - m) for x in xs]) or 1e-9


def robust_scale(vals):
    return max(1e-9, 1.4826 * mad(vals))


def dist(a, b, sm, sv):
    return math.hypot((a[0]-b[0])/sm, (a[1]-b[1])/sv)


def main():
    df = fetch(10000)  # fetch every available draw
    rows = []
    for _, r in df.iterrows():
        nums = [int(r[f"n{i}"]) for i in range(1, 8)]
        rows.append((int(r["round"]), state(nums)))

    residuals = []
    scored = []
    past_errors = []

    for i in range(MIN_HISTORY, len(rows)-1):
        # At T=i, only transitions ending at or before T are available.
        hist_states = [s for _, s in rows[:i+1]]
        sm = robust_scale([s[0] for s in hist_states])
        sv = robust_scale([s[1] for s in hist_states])
        cur = rows[i][1]

        # Analogues j must have a known j->j+1 transition by T.
        candidates = []
        for j in range(0, i):
            candidates.append((dist(cur, rows[j][1], sm, sv), j))
        near = [j for _, j in sorted(candidates)[:K]]

        # Predict only the ordinary next-State tendency of similar prior States.
        dmu = med([rows[j+1][1][0] - rows[j][1][0] for j in near])
        dvar = med([rows[j+1][1][1] - rows[j][1][1] for j in near])
        pred = (cur[0] + dmu, cur[1] + dvar)
        actual = rows[i+1][1]
        err = dist(pred, actual, sm, sv)

        # Threshold is generated only from earlier realized errors; no fixed % quota.
        flag = False
        threshold = None
        if len(past_errors) >= 30:
            center = med(past_errors)
            sigma = robust_scale(past_errors)
            threshold = center + ROBUST_Z * sigma
            flag = err > threshold

        item = {
            "t": rows[i][0], "target": rows[i+1][0],
            "mu_t": cur[0], "var_t": cur[1],
            "mu_actual": actual[0], "var_actual": actual[1],
            "mu_pred": pred[0], "var_pred": pred[1],
            "error": err, "threshold": threshold, "residual": flag,
        }
        scored.append(item)
        if flag:
            residuals.append(item)
        past_errors.append(err)

    eligible = [x for x in scored if x["threshold"] is not None]
    print("LOTO7 RESIDUAL WALK / observation only")
    print(f"draws={len(rows)} rounds={rows[0][0]}..{rows[-1][0]} eligible={len(eligible)}")
    print(f"state=(centroid, variance), analogues K={K}, residual threshold=prior median error + {ROBUST_Z}*robust_sigma")
    print(f"residuals={len(residuals)} rate={(len(residuals)/len(eligible)*100 if eligible else 0):.2f}%")
    if residuals:
        gaps = [b["target"]-a["target"] for a,b in zip(residuals,residuals[1:])]
        print(f"gap median={med(gaps) if gaps else 'NA'} min={min(gaps) if gaps else 'NA'} max={max(gaps) if gaps else 'NA'}")
        print("RESIDUAL TARGETS")
        for x in residuals:
            print(f"{x['t']}->{x['target']} err={x['error']:.3f} threshold={x['threshold']:.3f} mu {x['mu_t']:.2f}->{x['mu_actual']:.2f} var {x['var_t']:.1f}->{x['var_actual']:.1f}")


if __name__ == "__main__":
    main()
