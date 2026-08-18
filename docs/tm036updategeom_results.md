# TM.0.36.UPDATEGEOM results

Post-write ACT-query update-geometry wall after TM035. Product **0.0.004**.

## Decision: `updategeom_oracle_only`

On the two diagnostic worlds (both `reg1`), native sequential credit, Jacobi batch, and constraint-preserving scaled updates all fail the fix gate. A nearby feasible \(W\) exists and restores stored-row margins plus live 8/8. The local `_apply_act_query_update` optimizer is insufficient. **v39 is not frozen.**

No neural edit. v38 unchanged. 16-pass cap unchanged. 44 not installed.

- worlds: 8 diagnostic 2 native-already 6
- diagnostic routes: both `oracle_only`
- v39 freeze: **false**

git_head `f27b6188…` (freeze). Frozen runner SHA `fdcd11d3…`. Clean tree at DEV start.

## Diagnostic (`reg1`, both orders)

Parent store is at zero after credits 0–6. `write_only` leaves protected rows intact (`viol=1` is the new slot). Interference has 32 negative off-diagonals; min \(\Delta\)margin \(\approx -0.80\).

| arm | store | live | notes |
|-----|-------|------|-------|
| native | viol=3 slots 0/4/6 | 6/8 | 122 updates, budget exhausted (TM035 complete) |
| jacobi | viol=4 slots 1/3/5 | 4/8 | 61 bundled updates, worse than sequential; not path-dependence rescue |
| protect | viol=1 (new row) | 7/8 | protected rows intact; 17 scaled updates; **not acquired** |
| oracle | viol=0 | 8/8 | feasible; \(\gamma^*\approx 0.052\); \(\|W-W_0\|_F\approx 0.028\) |

Protect kept established rows and never fully blocked (\(\alpha=0\) count 0) but could not acquire the new row inside 16 passes. That is retention without acquisition, not silent scaled-learning success.

## Non-diagnostic

`reg0`/`reg2`: native already fixes (13 and 7 updates). Jacobi still collapses odd slots. `reg3`: post-write already 8/8; all intervention arms are no-ops except oracle which still reports a tiny feasible move.

Oracle \(\gamma^*\) is above 0.01 on every world. This is not a representation/capacity wall. TM032’s reachable mapping is consistent with a nearby feasible \(W\).

## Mechanism routing

Sequential order is not the lever (Jacobi is worse). Protect-every-update is not sufficient to acquire. The same update rule can leave a good partial solution and still sit a short Frobenius step from a jointly feasible \(W\). Next is a consolidation/optimizer redesign, not another replay budget, timing controller, or one-shot-only v39. Product **0.0.004**.
