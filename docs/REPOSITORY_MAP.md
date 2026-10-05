# REPOSITORY MAP｜2026-10-05

## 1. Active / 現役

### Model / runtime
- `lotocore.py`
- `x_agent.py`
- `fetch_loto7.py`
- `walk_forward.py`
- `random_kun.py`

### LOTO7 evaluation / Phase10
- `coverage_lab/`
- `compare_loto7_core_uniform_random.py`
- `compare_loto7_mesh.py`
- `observe_loto7_core_x_relation.py`
- `.github/workflows/experiment.yml`
- `.github/workflows/candidate-compression-phase10.yml`
- `.github/workflows/loto7-phase10-next-snapshot.yml`
- `.github/workflows/loto7-core-vs-random.yml`

### LOTO6 parallel line
- `loto6_*.py`
- `check_loto6_*.py`
- `run_loto6_x.py`
- corresponding `.github/workflows/loto6-*.yml`
- `.github/workflows/random-kun.yml`

LOTO6 is preserved as a parallel line and must not be merged conceptually with the current LOTO7 project state.

## 2. Archive / 過去探索

- `archive/loto7/legacy-probes/72-28/`
- `archive/loto7/legacy-probes/slow-fast/`
- `archive/loto7/legacy-probes/survivor-reentry/`
- `archive/loto7/legacy-probes/residual-play/`
- `archive/loto7/legacy-probes/allocator/`
- `archive/loto7/legacy-probes/tplus1/`
- `archive/workflows/`

Archive is not deletion. It is for reproduction and historical inspection, not current authority.

## 3. Frozen naming warning / Frozen命名衝突

This repository historically used "frozen" in multiple meanings.

1. old slow/fast probe: alpha selected on train then fixed for holdout.
2. coverage_lab frozen Round Packet: packet fixed before result reveal.
3. current project Frozen v0.1: 4-draw window / 30 features / past-only Z / K30 / 9 states / Branch3.

1 and 2 are NOT 3.

## 4. .flow/current.json

The stored Phase10 practical 10-ticket state is a repository-local historical current.
It is not the latest project-wide coordinate.

Current project research around Branch10 / Frozen v0.1 must be re-entered from its external authority and must not be inferred from legacy repo names.

## 5. Current repo reading order / 再入場順

1. `README.md`
2. `docs/REPOSITORY_MAP.md`
3. `.flow/current.json`
4. `.flow/decisions.md`
5. only then follow the route needed for the task

Do not read `archive/` as current state unless explicitly investigating history.
