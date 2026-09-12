import math
import random
from pathlib import Path

import pandas as pd

SEED = 20260912
SIMS = 1000
SRC = Path('data/loto6.csv')
OUT = Path('results/loto6_shape_direction_vs_shuffle.csv')


def load_shapes():
    df = pd.read_csv(SRC)
    nums = [c for c in df.columns if c.lower().startswith('n')][:6]
    if len(nums) < 6:
        nums = list(df.columns[1:7])
    shapes = []
    rounds = []
    for _, row in df.iterrows():
        xs = sorted(float(row[c]) for c in nums)
        m = sum(xs) / 6.0
        shapes.append([x - m for x in xs])
        rounds.append(int(row[df.columns[0]]))
    return rounds, shapes


def vec_sub(a, b):
    return [x - y for x, y in zip(a, b)]


def norm(v):
    return math.sqrt(sum(x*x for x in v))


def cos_angle(u, v):
    nu, nv = norm(u), norm(v)
    if nu == 0 or nv == 0:
        return None
    c = sum(x*y for x, y in zip(u, v)) / (nu * nv)
    return max(-1.0, min(1.0, c))


def metrics(shapes):
    cosines = []
    for i in range(1, len(shapes)-1):
        u = vec_sub(shapes[i], shapes[i-1])
        v = vec_sub(shapes[i+1], shapes[i])
        c = cos_angle(u, v)
        if c is not None:
            cosines.append(c)
    if not cosines:
        return {}
    ordered = sorted(cosines)
    n = len(ordered)
    mean_cos = sum(cosines)/n
    mean_angle = sum(math.degrees(math.acos(c)) for c in cosines)/n
    continuation = sum(c > 0.5 for c in cosines)/n      # angle < 60°
    reversal = sum(c < -0.5 for c in cosines)/n        # angle > 120°
    same_hemi = sum(c > 0 for c in cosines)/n
    reverse_hemi = sum(c < 0 for c in cosines)/n
    return {
        'mean_cosine': mean_cos,
        'mean_turn_angle_deg': mean_angle,
        'continuation_rate_cos_gt_0_5': continuation,
        'reversal_rate_cos_lt_-0_5': reversal,
        'same_direction_hemisphere_rate': same_hemi,
        'reverse_direction_hemisphere_rate': reverse_hemi,
    }


def percentile(xs, x):
    return sum(v <= x for v in xs)/len(xs)


def two_sided_p(xs, x):
    p = percentile(xs, x)
    return min(1.0, 2*min(p, 1-p))


def main():
    rounds, shapes = load_shapes()
    real = metrics(shapes)
    rng = random.Random(SEED)
    sims = {k: [] for k in real}
    for _ in range(SIMS):
        shuf = shapes[:]
        rng.shuffle(shuf)
        m = metrics(shuf)
        for k, v in m.items():
            sims[k].append(v)

    rows = []
    print('=== LOTO6 SIX-POINT SHAPE DIRECTION BUNDLES vs SHUFFLED ORDER ===')
    print(f'rounds={rounds[0]}..{rounds[-1]} sims={SIMS} seed={SEED}')
    print('bundle = two consecutive 6D shape-change vectors; compare their turning angle')
    print('null = same 400 observed shapes, temporal order randomly shuffled')
    for k, rv in real.items():
        xs = sims[k]
        sm = sum(xs)/len(xs)
        p05 = sorted(xs)[int(0.05*(len(xs)-1))]
        p95 = sorted(xs)[int(0.95*(len(xs)-1))]
        pct = percentile(xs, rv)
        p2 = two_sided_p(xs, rv)
        print(f'{k}: real={rv:.6f} shuffle_mean={sm:.6f} p05={p05:.6f} p95={p95:.6f} two_sided_p={p2:.3f} percentile={pct:.3f}')
        rows.append({'metric':k,'real':rv,'shuffle_mean':sm,'p05':p05,'p95':p95,'two_sided_p':p2,'percentile':pct})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f'saved -> {OUT}')
    print('LOTO6_SHAPE_DIRECTION_COMPLETE')


if __name__ == '__main__':
    main()
