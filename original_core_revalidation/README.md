# LOTO7 Original CORE Revalidation v0

Parent question: does the original CORE map (`center`, `variance`) place the next-draw state region better than a same-width random region, using only past data?

## Frozen protocol

- Latest 100 rounds only: 598..697 for the current 697-round input.
- State: population mean and population variance of the seven main numbers.
- Reference: K=20 nearest past states in past-only z-scored Euclidean distance.
- Prediction region: 25th..75th percentile rectangle of the successors of those K neighbors.
- Primary null: same-width rectangles anchored at randomly sampled past states. This preserves the non-uniform density of the observed state space.
- Secondary null: same-width rectangles placed uniformly inside past-observed bounds.
- No ticket generation, Frozen, Branch, Chance Timing, Portfolio, Mesh, Assembler, or Pair60.
- No parameter search after observing results.

## Leakage gates

The test script verifies that changing the target draw or any later draw does not change the prediction, results are deterministic, and every neighbor successor is known before the target.

## Current run

Input CSV SHA256: `2eb2de867369c1444754072b6d694c090b7548d51edad6fff88cbd11d3e0753b`

Primary result over rounds 598..697:

- CORE region hits: 19/100
- empirical-density matched same-width null: mean 12.8782/100, median 13, 5-95% 8..18
- empirical null P(random >= CORE): 0.0436 over 5,000 repeats
- difference vs empirical null mean: +6.1218 hits / +6.1218 percentage points

Secondary uniform-box null mean: 8.6574/100. It is not the primary comparator because uniform placement can under-represent dense central state regions.

## Boundary

This is one fixed 100-round probe. It is evidence that the original CORE region placement is a candidate for information at the state-region layer, not evidence of ticket-level advantage or future profitability. Do not tune K, quantiles, features, or subperiods from this result and reconnect them to this run as if pre-specified.
