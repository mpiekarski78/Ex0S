# TM.0.24.DISCRIMMAP DEV

Decision: **d1_fails_robustly**. D1 robust: **False**. D3 robust: **False**.

Next: `address_geometry_insufficient_no_competitive_linear_amendment`. SCORE unopened. No neural candidate. 1536 eligibility budget stays closed. Product **0.0.004**. `earned_next=false`.

## What was measured

280 cells on unused `TM024.DISCRIMMAP.DEV.` / `TWIN.`: 240 rank (4 addresses × 5 arms × 3 cue counts × 2 orders × 2 worlds) plus 40 twins (2-cue, both orders). Same organism trajectories as ELIGMAP capture. Training used select-tick addresses only.

Normalized geometric margin floor is 0.01, frozen before DEV. D4 is a diagnostic ceiling only.

## Result

The batch max-margin linear oracle **cannot fit eight-cue training addresses**. All 16 required D1 8-cue rank cells fail training sign, training margin, probe ranking, probe margin, and perturbation. D3 and D4 also fail every 8-cue rank cell. D0 (nearest-prototype control) fails 8-cue as expected.

Two-cue is weaker evidence, not a pass of the ladder: D1 fits all 16 two-cue training cells and fully passes 8/16 rank plus 4/8 twins. Four-cue D1 can often fit training (12/16) but does not pass the frozen probe/margin/perturbation gate (0/16).

Pairwise COLLISIONMAP distinction is still not robust linear separability. This package does not install D1–D4, does not declare v31/v32, and does not open SCORE.
