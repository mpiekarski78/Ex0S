# TM.0.24.LINEAGE results

Product remains **0.0.004**. `earned_next=false`. `ex0s=null`. `eligible_for_000005=false`.

## What was frozen

- Phase A scientific floors (`τ_k`, `δ_B`, `δ_P`, CI method, DEV stream, QUAL/EVAL commitments) were not moved.
- Phase 0B pinned engine SHAs in [`lineage_engine.preflight.lock`](lineage_engine.preflight.lock).
- [`lineage_engine.candidate.lock`](lineage_engine.candidate.lock) equals those SHAs.
- Compute amendment reduced scope to P=16, B=2, W=2 (not 128). Floors unchanged.

## Scored run

Five generations, P=8 (subset of the frozen P=16). `F_search` always. Last mean adult lower-quartile proxy: **0.115**.

Checkpoint champion on DEV triplet 0 (base replication):

| Panel | adult mean | birth mean | plasticity-off mean | CI lower | G_k | clear |
| --- | --- | --- | --- | --- | --- | --- |
| A | 0.088 | 0.000 | 0.388 | 0.00 | no | no |
| B | 0.125 | 0.000 | 0.013 | 0.00 | no | no |
| C | 0.100 | 0.000 | 0.075 | 0.00 | no | no |

τ_L0 = 0.60. Failed panels were not reused. Consolidation triplet was not opened. Arm C control mean F_search = 0.15.

## Claims that are refused

- L0 unlock, QUAL, EVAL, RC, `earned_next`, `eligible_for_000005`, and Ex0S 0.0.005.
- Arm C vs Arm D superiority (neither prospect is on `origin/main`).
- Absolute impossibility of L0. This is a **bounded** wall under the frozen architecture, search, data, and compute budget.

A negative at L0 is a primary TM.0.24 result. Next response is an architecture decision, not scorer weakening.
