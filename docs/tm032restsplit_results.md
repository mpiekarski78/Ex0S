# TM.0.32.RESTSPLIT results

Matched post-awake REST-component split on frozen v37. Product **0.0.004**.

## Decision: `restsplit_awake_rehearsal_sufficient`

v38 route: **adaptive violation-driven rehearsal**. Not v38 itself. `R` unmodified. `ACT_RECALL_MODES` untouched.

**Limitation:** routing evidence is **two informative splits, both `reg1`**. Six splits were already converged at post-awake and do not vote. Do **not** install the observed 44 row-updates as a fitted constant; that count is 16-pass work on those two splits, not a law.

- splits: 8
- diagnostic (`none` does not already fix): 2
- baseline already converged: 6
- diagnostic routes: both `awake_rehearsal_sufficient` (agree; not mixed)

Runner SHA `bd591d29…`. DEV git_head `b122cac` (freeze). DEV lock SHA `7c6a8833…`.

## All arm outcomes (routing selects one lever; all five remain visible)

Fix gate: `n_store_violations==0` and live ranking 8/8. Margin recorded, not a gate. Parent snapshot unchanged on every split.

### Diagnostic splits (reg1, both orders)

`none` fails (3 store violations, live 6/8). `mix_only` does not fix. `awake_only`, `replay_no_mix`, and `full_rest` each fix (44 gated updates, first-converged pass 16). Frozen priority selects **awake_only** when several compute-matched arms also fix.

| split | none | awake_only | replay_no_mix | mix_only | full_rest |
|-------|------|------------|---------------|----------|-----------|
| A_then_B\|reg1 | fail (viol=3, 6/8) | fix (upd=44) | fix (upd=44) | fail (viol=3, 6/8) | fix (upd=44) |
| B_then_A\|reg1 | fail (viol=3, 6/8) | fix (upd=44) | fix (upd=44) | fail (viol=3, 6/8) | fix (upd=44) |

### Baseline-converged splits

`none` already 8/8 with zero store violations on reg0, reg2, and reg3 (both orders). Sibling arms remain fixed; they do not vote.

| split | none | awake_only | replay_no_mix | mix_only | full_rest |
|-------|------|------------|---------------|----------|-----------|
| A_then_B\|reg0 | fix | fix | fix | fix | fix |
| B_then_A\|reg0 | fix | fix | fix | fix | fix |
| A_then_B\|reg2 | fix | fix | fix | fix | fix |
| B_then_A\|reg2 | fix | fix | fix | fix | fix |
| A_then_B\|reg3 | fix | fix | fix | fix | fix |
| B_then_A\|reg3 | fix | fix | fix | fix | fix |

Slow mix alone never repaired a diagnostic split. Full REST repaired them, but so did compute-matched awake rehearsal without mix or REST wrapper.
