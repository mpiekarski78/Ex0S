# TM.0.12.CONTEXT results: representation audit of Ex0S 0.0.003

**Ex0S under test:** **0.0.003** (not a new stamp)  
**Date:** 15 August 2026  
**Regime:** representation / provenance audit (not a recipe jump)  
**Run:** `runs/2026-08-15_175049_tm012context`  
**Locks:** [`genome_011.lock`](genome_011.lock) (organism) · [`context_012.lock`](context_012.lock) (apparatus, preregistered)

`earned_next`: **false** — no Ex0S 0.0.004. No genome patch. Do not pre-name a future stamp Abstraction.

## Headline

| Bucket | Result |
|--------|--------|
| Within-model (C0, C1) | **2/2** pass |
| Expected boundaries (C2–C7) | **7/7** confirmed |
| Unexpected Fail | **0** |
| Resource Boundary | **0** |
| C7 indistinguishability witness | same frozen motors · context expects differ · both confirmed |
| Genome / apparatus interventions | **0 / 0** |

## Hypotheses

| Id | Candidate | Role |
|----|-----------|------|
| H0 | bare `Y` | baseline (frozen compose) |
| H1 | `(Y, here)` | structured |
| H2 | `(Y, pred)` | structured |
| H3 | `(Y, origin/path)` | structured |
| H4 | `(Y, fact_id)` | diagnostic upper bound only — not a product claim |

## Elimination map

| Cell | Class | Discriminates | Frozen motors | Context expects |
|------|-------|---------------|---------------|-----------------|
| c0_unique | within_model_pass | apparatus intact | path-correct | = frozen |
| c1_benign_reuse | within_model_pass | reuse not pathological | same motor both cues | = frozen |
| c2_here_split | expected_boundary | H1 vs H0 | bind-only (strong wrong-here) | cue_z wants here-filter |
| c3_pred_split | expected_boundary | H2 vs H1 | both cues → strong | preds differ; here identical |
| c4_pred_collision | expected_boundary (hold-out) | H3 vs H2 | both → strong | origins differ; pred A shared |
| c5_path_depth | expected_boundary (hold-out) | path history | both → strong | short vs long path |
| c6_evidence_trap | expected_boundary | evidence ≠ provenance | strong wrong wins | cue_x wants weak right-here |
| c7_a / c7_b | expected_boundary (hold-out) | structural | both HOLD | PRESS vs TUNE |

**Reading:** bare-token compose (H0) is sufficient for C0/C1. For every preregistered ambiguity (C2–C7), frozen 0.0.003 collapses to a single frontier decision that cannot track path-dependent continuations. C7 shows the failure is **structural**: two required CONTEXT outputs map to identical frozen state.

This pass does **not** yet rank H1–H3 sufficiency for a future genome — see **TM.0.12.MINIMAP** ([`tm012minimap_results.md`](tm012minimap_results.md)): among preregistered candidates, `(token, origin)` is the least-structured sufficient representation for C2–C6; C7 is unobservable from provenance.

## Minimality (apply later)

When proposing a CONTEXT representation, report which of C2–C7 each of these solves:

- bare token
- `(token, here)`
- `(token, predecessor)`
- `(token, path-origin)`
- opaque fact_id (upper bound only)

Jump the genome only if the **smallest additional structured state** covers all preregistered ambiguity classes.

## Deferred

- **LOOKAHEAD** — separate battery; do not mix (can brute-force continuations).
- **\|S\| latency** — [`tm011bound_perf.md`](tm011bound_perf.md); behavioral-equivalence regression required.

## Reproduce

```bash
python -m experiments.run_tm012context --write-lock   # once, before hold-outs
python -m experiments.run_tm012context --seed 12345 --workers 4
python tests/test_tm012context.py
```
