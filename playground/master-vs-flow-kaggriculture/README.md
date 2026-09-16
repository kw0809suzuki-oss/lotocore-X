# Master vs Flow — Kaggriculture One Shot

## Purpose
A deliberately isolated playground for one one-shot match between the two policies that emerged from the 30-day text battle.

**Objective:** Day30 terminal money only.

This is play, not a new LOTO model and not a modification of the main Kaggriculture agent.

## Isolation rule
- Everything for this experiment stays under `playground/master-vs-flow-kaggriculture/` on branch `play/master-vs-flow-kaggriculture`.
- Do not import from or modify LOTOCORE-X model logic.
- Do not merge findings back into LOTO automatically.
- Do not modify the existing Kaggriculture agent from here.
- Knowledge may be recorded, but transfer to another project requires a separate explicit decision.

## Frozen policies
### MASTER
Concentrate on WHEAT production capacity early, build inventory, stop expansion around the late-middle phase, harvest, then progressively liquidate inventory toward Day30.

Text-battle liquidation shape: `10 -> 15 -> 15 -> 20 -> 25 -> 25 -> 30 -> final liquidation`.

### FLOW
Preserve liquidity early, do not chase MASTER's WHEAT capacity race, move to the best available non-WHEAT revenue path, collapse back to the most terminal-efficient single path, stop new investment earlier, and begin liquidation earlier.

## One-shot rule
1. Same game rules.
2. Same initial conditions.
3. Same seed/environment where supported.
4. No tuning after seeing the opponent or result.
5. No rematch for optimization.
6. Winner is determined only by actual Day30 terminal money.
7. If the real engine cannot express a text-battle action, record the mapping instead of silently inventing behavior.

## Epistemic rule
Do not call either policy better before an actual executable mapping and result exist.

## Next step
Locate the authoritative Kaggriculture game runner/rules, write an explicit mapping from each frozen policy to legal actions, then run exactly one match and save the raw result here.
