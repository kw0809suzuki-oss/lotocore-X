# LOTO6 Future Protocol v1

Frozen before future evidence begins.

## Purpose

Use historical LOTO6 results only as development material, then test one frozen probabilistic machine on future draws.

- Development region: rounds 1-2139
- Future evidence region: round 2140 onward
- Maximum evaluation horizon: 200 future draws
- No future result may be used to change v1.

This protocol tests predictive information, not profitability and not ticket recommendations.

## Official outcome space

LOTO6 outcome space is all unordered 6-number combinations from 1..43.

- Total combinations: C(43,6) = 6,096,454
- Primary null: uniform joint distribution
- Uniform probability per combination: 1 / 6,096,454

## Frozen predictor

Version: `frequency-joint-v1`

For target round t:

1. Use only draws strictly before t.
2. Count historical occurrences c_i for each number i in 1..43.
3. Apply Laplace smoothing with alpha = 1.
4. Define per-number weight

   w_i = (c_i + 1) / (6N + 43)

   where N is the number of prior draws.
5. For any legal 6-number combination S, define unnormalized joint weight

   q(S) = product of w_i for i in S.
6. Normalize over all legal 6-number combinations:

   P_v1(S) = q(S) / Z

   where Z is the degree-6 elementary symmetric polynomial of the 43 weights.

This produces a proper joint distribution, assigns positive probability to every legal outcome, and respects the no-duplicate six-number constraint.

Shape, Position, Transition, State-conditioned Random, and Bundle structure are not inputs to v1.

## Primary score

For realized future outcome Y_t:

logLR_t = log P_v1(Y_t | history before t) - log P_uniform(Y_t)

Cumulative evidence:

C_t = sum logLR_t

Only this cumulative log likelihood ratio is used for the primary gate.

## Gate and stopping rule

- PASS if cumulative log likelihood ratio >= log(100) ~= 4.605170.
- Evaluation starts at round 2140.
- Maximum horizon is 200 future draws.
- If PASS is not reached by the end of the 200-draw horizon, v1 status is NOT ESTABLISHED.
- NOT ESTABLISHED is not a proof of mathematical independence.

No alternate metric may rescue v1.

Brier score, ticket hits, State transitions, bundle coverage, entropy, or visually favorable subperiods may be reported only as diagnostics and may not change the PASS gate.

## Change control

After future evidence begins, changing any of the following invalidates v1:

- model formula
- alpha
- start round
- outcome space
- primary null
- primary score
- PASS threshold
- 200-draw horizon
- leakage rules

New ideas must be versioned separately as v2 or later and must not alter the v1 record.

## Evidence record

For every future round, record before evaluation:

- target round
- model version
- last history round used
- model parameters
- prediction artifact hash or equivalent immutable identifier

After the draw, append:

- realized six-number combination
- model probability for that combination
- uniform probability
- model log score
- uniform log score
- per-round logLR
- cumulative logLR

The evidence ledger must be append-only in interpretation: prior predictions and scores are never rewritten because later results are inconvenient.

## One-line definition

Past data is for development. Future draws are for evidence.
