# TM.0.11.FAMILY results: frozen COMPOSE vs composition worlds

**Ex0S:** **0.0.003 — Frozen Composition**  
**Date:** 15 August 2026  
**Regime:** freeze (not a recipe jump)  
**Run:** `runs/2026-08-15_154037_tm011family`  
**Lock:** [`genome_011.lock`](genome_011.lock) (organism + preregistered E–G generators + scorer + seed list)

## Headline

| Split | Solved | Max depth |
|-------|--------|-----------|
| All | **252/252** | — |
| Developed A–D | **144/144** | 3 |
| Hold-out E–G | **108/108** | **4** |
| Genome changes | **0** | — |
| Apparatus interventions | **0** | — |

## Per family

| Family | Split | Depth | Solved |
|--------|-------|-------|--------|
| A | developed | 2-hop | 36/36 |
| B | developed | 3-hop | 36/36 |
| C | developed | junk off-path (wrong motor) | 36/36 |
| D | developed | first-hop branch | 36/36 |
| E | hold-out | 4-hop | 36/36 |
| F | hold-out | first-hop tie/break | 36/36 |
| G | hold-out | revise-downstream | 36/36 |

All independent measures **1.000** (`compose_depth`, `no_transitive_shortcuts`, `match_drops_junk`, `evidence_branch`, `tie_hold`, `revise_downstream`, `upstream_stability`, `reset_continuity`, `s_necessity`, `permutation_invariance`, `genome_delta`).

## What this shows

Factorized learned knowledge generalizes under a frozen compositional mechanism across unseen depth, first-hop branching, and downstream revision.

- **E:** developed max depth 3; hold-out depth 4; no hop flag added.
- **D/F:** first-hop evidence only — stronger `X→Y` vs `X→Z`; equal first-hop → HOLD. Downstream-stronger traps did not win (no lookahead).
- **G:** revise only `Y→PRESS` / `Y→TUNE`; `hash(X→Y)` stable; no `X→motor` shortcut; cue X flips motor without relearning X.
- **C:** high-support junk uses a **different** motor than the chain; MATCH must drop it or the probe Fails.

## Audit notes (apparatus)

Before the recorded run above:

1. Family C junk initially shared the chain motor (steal invisible). Fixed to wrong-motor junk; **developed** only — hold-out E–G generators unchanged (lock hashes intact).
2. `earned_frozen_composition` / `ex0s: 0.0.003` now requires the full default battery (seed 12345 × 12 × 3 = 252). Smoke runs no longer stamp 0.0.003.

## What this does not show

- Planning / lookahead over downstream evidence
- English / motor-bar progress (B Fail untouched)
- New epistemic machinery (confidence, decay, hop caps)

## Reproduce

```bash
python -m experiments.run_tm011family --seed 12345 --per-family 12 --births 3
```

Freeze verify reads `docs/genome_011.lock`. Hold-out generator SHAs must match; peek-then-edit voids the hold-out.
