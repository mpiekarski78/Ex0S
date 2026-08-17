# CORTEX v32 implementation erratum

Authorized after scored TWOSCALE DEV on `0be51e8`. Not a new law. Not an eight-cue solve.

Historical [`lineage_twoscale.dev.lock`](lineage_twoscale.dev.lock) first-match remains **architectural_wall_acquire**. Four-cue acquire/stability/reversal/specificity passed. Eight-cue acquire ranking is 7/8. This erratum does not rewrite that lock, the decision, or the addendum. Product **0.0.004**. `earned_next=false`. `ex0s=null`. Lineage stays closed.

Ancestor live neural SHA at freeze: `41fed1e7a651a7570dc1aa869ad648cf30e2a8cb2efcf3eab315cb748e3d392e`.

Amendment item 2 already says: score and credit ACT from `last_p1` **when it is set**. `W_op` / `W_pred` keep action-owned delayed credit on live post-motor `ρ_elig`. This package repairs three implementation mismatches with that scored law. It does not authorize competitive/heterosynaptic ACT plasticity.

## Authorized corrections (implement after this freeze)

1. **Tie consistency.** `_unique_act_winner` must use the same `1e-12` band as `_choose_actuator`. A near-tie or all-near-zero scores is not a unique winner. Selection may still RNG-pick among ties. Runner `unique_winner` (exact equality) stays the frozen measurement convention.

2. **P1 eligibility gating, with missing-P1 fallback.** Query-mode ACT:
   - `rho_p1` present and energetic → gate and learn from P1, even if `rho_motor` is zero.
   - `rho_p1` present but zero → no ACT query credit, even if `rho_motor` is energetic.
   - `rho_p1` absent → preserve legacy `rho_motor` fallback and its `elig_motor` gate (older pending checkpoints).
   Removing the fallback is a behavioral amendment, not this erratum. EMIT stays on `elig_motor` / `rho_motor`. `W_op` / `W_pred` stay on `elig_op`. W1 stays off.

3. **Checkpoint continuity.** Persist and restore `dev_epoch` and episode/REST counters. Missing keys default to zero so pre-erratum snapshots still load.

## Procedure after this freeze is on origin/main

Implement the three corrections. Freeze [`lineage_twoscale.compat.runner.lock`](lineage_twoscale.compat.runner.lock) on origin/main before any 36-cell replay. Then one deterministic comparison against the historical DEV lock without rewriting it. Compatibility lock if the complete semantic payload matches; otherwise a permanent mismatch lock, then unused `TM025.TWOSCALE.R3.DEV.` / `TWIN.`.

## Refuse

Competitive/heterosynaptic rival depression; another solver MAP; rewriting historical TWOSCALE/v32/AFFINEMAP locks; removing the missing-P1 fallback; executing replay before the compat runner freeze; treating this erratum as an eight-cue solve; SCORE; W1; A3; increase n; `earned_next`; 0.0.005.
