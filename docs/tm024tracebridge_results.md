# TM.0.24.TRACEBRIDGE DEV

Decision: **p1_not_usable_by_online_class**.

Arm robust: B0 **false**, B1 **false**, B2 **false**, B3 **false**, B4 **true**.

Carrying P1 to scoring and credit is enough for the frozen D1 oracle and not enough for the tested online class. Write-geometry closed. SCORE unopened. No neural candidate. 512/1536 budgets stay closed. Product **0.0.004**. `earned_next=false`.

## What was measured

85 cells on unused `TM024.TRACEBRIDGE.DEV.` / `TWIN.`. Same delayed select→credit organism. Default `act_score_mode=query`. W1 not resurrected.

| Arm | Address | Learner | 8-cue rank | 4-cue | 2-cue / twin | Eco / HOLD / REST |
| --- | --- | --- | --- | --- | --- | --- |
| B0 | P5 | v29 `W_act_query` | 0/4 | 0/4 | 0/4 / 0/2 | 1 / 1 / 0 |
| B1 | exact P1 | v29 | 0/4 | 0/4 | 0/4 / 0/2 | 1 / 1 / 0 |
| B2 | v_end register λ=0 | v29 | 0/4 | 0/4 | 0/4 / 0/2 | 1 / 1 / 0 |
| B3 | P1 | D3 competitive, signed by adv | 0/4 | **4/4** | **4/4 / 2/2** | **1 / 1 / 1** |
| B4 | P1 | D1 hard-margin ceiling | **4/4** | **4/4** | **4/4 / 2/2** | 0* / 1 / 1 |

\*B4 ecological is not required. D1 classifies handles, not reversal of one cue.

B4 eight-cue train γ ≈ 0.0398, probe γ ≈ 0.0247, perturbation stable. That matches PHASEMAP P1.

B1 equals B2 (λ=0 register of the same `v_end` tick). Both fail even two-cue ranking: last write wins, large margin on the wrong handle, perturbation stable on that wrong handle. Single-cue ecological reversal still passes, as expected for last-write.

B3 learns opposing 2-cue and 4-cue maps, including twins, REST, and ecological reversal. It does not train or rank at eight cues.

HOLD: `W_act_query` is bitwise unchanged under HOLD clamp on every arm.

## What this does not authorize

Do not install a 512-row event-end trace. Do not change the v29 write law. Do not resurrect W1. Do not declare v31/v32. Do not open SCORE. P1 is oracle-separable and not usable by sequential outer-product or by the frozen D3 online rule at eight cues. A later package would need a different learning class, not a retry of this bridge.
