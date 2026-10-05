# LOTO7 Candidate Compression Lab

Purpose: measure how far a model-derived candidate space can be compressed before useful information is lost.

This is not the current Branch10 / Frozen v0.1 authority.
This lab belongs to the repository-local Loto Core / X / Phase10 line.

## Active Phase10 surface / 現役

- `fixed_k_curve.py`
- `state_observables.py`
- `practical_10ticket_backtest.py`
- `current_10ticket_snapshot.py`
- `regression_check.py`
- `generate_round_packets.py`
- `validate_round_packet.py`
- `round_packet.schema.json`

Current flow:

history -> model scores/ranks -> pre-result fixed Round Packet -> choose K -> fixed allocator -> 10 tickets -> reveal actual -> evaluate.

The phrase **frozen Round Packet** means only "fixed before result reveal".
It is NOT the current project model named **Frozen v0.1**.

## Historical evaluation helpers / 過去Phaseの観測補助

Kept for reproducibility, not current authority:

- `adaptive_k_strict_forward.py`
- `freshness_rule_forward_test.py`
- `gate_freshness_scan.py`
- `multiblock_state_validation.py`
- `strength_bins.py`
- `temporal_state_validation.py`

Allocator selector/veto probes were moved to `archive/loto7/legacy-probes/allocator/`.

## Terms
- Candidate Span K: number of unique LOTO7 numbers retained before ticket allocation.
- Candidate Compression Ratio: K / 37.
- Compression Tolerance: how performance changes as K decreases while model output and allocator remain otherwise fixed.

## Boundary
LOTO CORE / X remain the model layer. This lab sits after model scoring/ranking and before the existing ticket allocator.

Never choose K using the target round result.
