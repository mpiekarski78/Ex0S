# TM.0.25.TWOSCALE DEV

Decision: **architectural_wall_acquire**.

v32 two-timescale P1 memory plus slow consolidation. Unused `TM025.TWOSCALE.DEV.` / `TWIN.`. SCORE unopened. Product **0.0.004**. `earned_next=false`. Lineage stays closed.

## What passed

| Phase | 2 cues | 4 cues | Notes |
| --- | --- | --- | --- |
| Acquire ranking | 4/4 | **4/4** | First online 4-cue acquire in this chain (AFFINEMAP A2 was 0/4) |
| Stability (rank ∧ γ≥0.01 ∧ pert after REST) | 4/4 | **4/4** | min γ 0.115 / **0.042** |
| Twin | 4/4 | | |
| Ecological reversal | **4/4** | | min γ ≈ 1.00; 2 contradictory replacements |
| Specificity | **4/4** | | min γ ≈ 0.993 |

n=64. Query scoring. Eight-slot store filled without match collapse (`n_episodes=8` on 8-cue cells). W1 off. A3 out.

## What failed

Eight-cue acquire ranking is **7/8** on every unused world/order. min γ = **−0.01609**. The miss is one cue on the wrong side of the two-actuator separator. REST replay (128 signed updates) does not recover the 8th (stable 8-cue also 7/8, min γ −0.02587).

It is not a 4-cue wall, not a reversal wall, not a specificity wall, not an empty-store failure, and not an intercept/A3 story.

## Escalation

R1 mixed slow `W_act_query` after every REST epoch (not the freeze). Acquire is pre-REST, so that could not explain 7/8. R2 mixed once per REST as specified; 8-cue acquire unchanged. The recorded DEV neural is that mix-once REST law (plus a no-op duplicate constant block, later stripped). A later unrecorded awake-store-replay pass collapsed four-cue ranking and is not the scored organism.

## Wall

**Eight-cue acquire ranking under two-timescale P1 outer-product consolidation.** Four-cue maps acquire, hold, reverse, and stay specific. Homogeneous D1 still ranks eight cues; this organism law does not. No further MAP. No SCORE. No `earned_next`.
