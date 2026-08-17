# TM.0.24.LIFECYCLEMARGINMAP DEV

Decision: **consolidation_margin_loss**.

112 unused cells on `TM024.LIFECYCLEMARGINMAP.DEV.` / `TWIN.`. Freeze commit `607b834`. Runner.py SHA `5f7dc1a7…` unchanged. SCORE unopened. No neural candidate. Product **0.0.004**. `earned_next=false`.

## First-match

M1 8-cue live min margin reached 0.01 before REST (0.02002) and fell below afterward (0.00017). That is the frozen consolidation rung. It is not `margin_conditioned_replay_supported`, not `replacement_and_margin_conditioned_replay_jointly_causal`, and not `max_margin_ceiling_only`.

M1 8-cue ranking was already incorrect on 3 / 8 probes before REST. The 0.02002 figure is the weakest live probe margin, not a four-phase pass. REST then collapsed that number. Stored P1 signs never all-correct (`first_all_correct_call=None`; 93 / 128 actual PA updates).

## Control reproduced

M0 error-only L2 8-cue cells match published R2 L2: 14 / 128 updates, first-all-correct at call 20, 108 idle calls, ranking correct, REST raised live min margin 0.00211 → 0.00643, still below 0.01, perturbation fail. REST helped the error-only boundary. It did not create the R2 miss.

M0 2-cue and 4-cue stability passed. M0 eco/spec passed with two live reversal updates. Eco/spec REST fields are JSON `null`.

## Other arms

| Arm | 8-cue acquire | 8-cue stable | eco | spec |
| --- | --- | --- | --- | --- |
| M0 error-only L2 | ranking pass | ranking pass, γ=0.00643 | pass | pass |
| M1 C3 PA L2 | ranking fail, γ=0.02002 | ranking fail, γ=0.00017 | pass | fail |
| M2 C3 PA L3 | ranking fail | fail | fail | fail |
| M3 D1 ceiling | pass | pass, γ≈1.24 | fail | pass |

M3 8-cue stability passes. M3 eco fails (no live reversal training; ceiling-only). So M3 is not four-phase and does not take the ceiling rung.

Phase flags: `{'M0': {'acquire_all': True, 'acquire8': True, 'twin': True, 'stable': False, 'plasticity': True, 'specificity': True}, 'M1': {'acquire_all': False, 'acquire8': False, 'twin': True, 'stable': False, 'plasticity': True, 'specificity': False}, 'M2': {'acquire_all': False, 'acquire8': False, 'twin': True, 'stable': False, 'plasticity': False, 'specificity': False}, 'M3': {'acquire_all': True, 'acquire8': True, 'twin': True, 'stable': True, 'plasticity': False, 'specificity': True}, 'm1_pre_rest_reaches_0.01': True, 'm1_post_rest_below_0.01': True, 'm1_never_reaches_0.01': False, 'ecological_match_not_first_failure': True}`.

Write-geometry closed. 512/1536 budgets stay closed. Same frozen DEV execution refused.
