# LOTO7 Candidate Compression Lab

Purpose: measure how far a model-derived candidate space can be compressed before useful information is lost.

This is not the LOTO6 Bundle Coverage problem and not T+1 Coverage.

## Terms
- Candidate Span K: number of unique LOTO7 numbers retained before ticket allocation.
- Candidate Compression Ratio: K / 37.
- Compression Tolerance: how performance changes as K decreases while model output and allocator remain otherwise fixed.

## Design boundary
LOTO CORE / X remain the model layer. This lab sits after model scoring/ranking and before the existing ticket allocator.

Required flow:
history -> model scores/ranks -> frozen Round Packet -> choose K -> fixed allocator -> 10 tickets -> reveal actual -> evaluate.

Never choose K using the target round result.

## Controls
1. model score + adaptive K
2. model score + fixed K
3. same K with within-pool random ordering
4. uniform random over 1..37
5. optional score-shuffle control

## Current Phase 1
Expose and freeze full 1..37 score/rank vectors without changing existing final 7-number behavior.
