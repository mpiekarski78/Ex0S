# TM.0.34.TEMPORALWALL results

Temporal placement wall after v38 close. Product **0.0.004**.

## Decision: `temporalwall_temporal_interference`

Exact frozen TM032 `awake_only` still rescues the diagnostic post-awake snapshots (44 updates, live 8/8). Acquisition still fails because later credits reintroduce store violations after the store had already been at zero. v38 unchanged. 16-pass cap unchanged. 44 not installed.

- worlds: 8 diagnostic 2 v37-already 6
- diagnostic routes: both `tm032_rescue_holds`
- rebreak traces: 12 (including both diagnostic v37/fixed/adaptive orders)
- cross-row break traces: 6 (reg0 only; not the diagnostic first-match)
- margin-without-count traces: 22 (secondary; not a v38 repair)

git_head `b781696d…` (freeze). Frozen runner SHA `f097b6e2…`. Clean tree at DEV start.

## Diagnostic (`reg1`, both orders)

v37 and fixed extra: credits 0–6 sit at **zero** store violations. Credit 7 (last cue) introduces slots **0, 4, 6** and spends 122 of 130 updates without clearing them.

Adaptive: zero through credit 4; credit 5 introduces 0/2/4 and plateau-stops; credit 6 returns to zero; credit 7 rebreaks including slot 6. Final even-slot collapse.

| cell | post-awake | TM032 awake_only |
|------|------------|------------------|
| A_then_B\|reg1 | fail (viol=3, 6/8) | fix (upd=44, 16 passes) |
| B_then_A\|reg1 | fail (viol=3, 6/8) | fix (upd=44, 16 passes) |

## Mechanism routing

Later credits rebreak consolidated rows → **temporal interference / protection**. Domain mismatch is not the next problem. Within-pass cross-row breaks exist on reg0 and margin-without-count is common; those are recorded, not a v38 patch and not a license to reuse 44.
