# TM.0.38.CONESPLIT results

Linearized vs exact-SOC vs joint-SOCP split after TM037. Product **0.0.004**.

## Decision: `conesplit_mixed_routes`

No neural edit. Oracle \(W^*\) is diagnostic only. No v40 freeze. No candidate lock.

Diagnostic = 16-cycle linearized Dykstra probe fails (6 of 8 worlds). Those six do **not** share one cause.

- worlds: 8 diagnostic 6 lin16-already 2
- routes: `budget_causal, budget_causal, joint_socp_only, joint_socp_only, geometry_and_budget, geometry_and_budget, lin16_already_converged, lin16_already_converged`

| replicate | route | lin_conv | soc_16 | soc_conv | oracle |
|-----------|--------|----------|--------|----------|--------|
| reg0 | budget_causal | 42 cycles, 8/8 | fail @16 | 48 cycles, 8/8 | 8/8 |
| reg1 (TM037 diagnostic) | joint_socp_only | fail @256, leftover 6 | fail @16 | fail @256, leftover 6 | 8/8, \(\gamma^*\approx 0.052\), \(\|W-W_0\|_F\approx 0.028\) |
| reg2 | geometry_and_budget | fail @256 | fail @16 | 226 cycles, 8/8 | 8/8 |
| reg3 | lin16 already | — | — | — | — |

Exact SOC never succeeds in 16 cycles. On the original TM037 world (reg1), neither linearized nor exact cyclic projection reaches the feasible set by 256; only joint \(W^*\) does. Other seeds show extra linearized cycles can suffice (reg0) or exact cones plus extra cycles (reg2). Degenerate \(W^\top d=0\) skips were 0.
