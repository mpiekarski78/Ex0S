# TM.0.33.ADAPTREHEARSE results

v38 adaptive violation-driven rehearsal. Product **0.0.004**.

## Decision: `adaptrehearse_core_acquire_fail`

Neither extra-replay arm restored eight-cue acquire on the diagnostic triples. The 16-pass REST allowance is insufficient, or the update rule hits a convergence wall. Do **not** install 44 row-updates. `R` unmodified. `ACT_RECALL_MODES` untouched.

**Limitation:** TM032 routing evidence was two `reg1` splits. TM033 diagnostic triples are the same two seed/order cells (`reg1`, both orders). Six triples were already v37-converged and do not vote.

Leftover debt after REST was **0**. REST repaid recorded awake pass debt (all ≤16). The leftover-debt integrity failclass did not fire; this outcome is not a lifecycle compute-conservation claim.

- triples: 8
- diagnostic: 2
- v37 already converged: 6
- budget-exhausted cells: 4 (v37 and `fixed_extra_replay` on both `reg1` orders)
- leftover debt after REST: 0
- diagnostic routes: both `both_fail` (agree; not mixed)

git_head `64883d5c…` (controller). Neural SHA `53227d84…`. Frozen runner SHA `91d8074d…`. Clean tree at DEV start.

## Diagnostic triples (`reg1`, both orders)

Fix gate: `n_store_violations==0` and live ranking 8/8.

| triple | v37_awake_cap | adaptive_violation | fixed_extra_replay |
|--------|---------------|--------------------|--------------------|
| A_then_B\|reg1 | fail (viol=3, 6/8, upd=130, 16-pass exhaust) | fail (viol=4, 4/8, upd=13, plateau) | fail (viol=3, 6/8, upd=130, 16-pass exhaust) |
| B_then_A\|reg1 | fail (viol=3, 6/8, upd=130, 16-pass exhaust) | fail (viol=4, 4/8, upd=13, plateau) | fail (viol=3, 6/8, upd=130, 16-pass exhaust) |

Adaptive stopped on plateau (violation count not decreasing) before the 16-pass cap. Fixed extra used the same violation-row targeting without plateau stop, exhausted the cap, and still failed. Targeting was `violation_rows` on every cell, so this is not a prioritization-versus-all-rows result.

## v37-already triples

`v37_awake_cap` already 8/8 with zero store violations on reg0, reg2, and reg3 (both orders). Those triples do not vote. Adaptive plateau-stopped short of the v37/fixed fix on some of those cells; that is recorded, not a diagnostic route.
