# CORTEX v12 statistical motor-learning contract

**Lab:** TM.0.23.CORTEX.V12.STAT  
**Product:** 0.0.004 · `earned_next=false` · `ex0s=null`  
**Authorized by:** [`cortex_diagnosis.v11.lock`](cortex_diagnosis.v11.lock)

Retain v11 D1 press+harm floors, population extras, swapped D2 HOLD window, sealed-seed worlds, and git-pinned reveal. Keep `holds>=5`, `beneficial>=3`, `rho_ok`. Do not lower `holds>=5`.

Authorized neural law (implement only after this apparatus is on `origin/main`):

When a credited ACT `body_adv` has the opposite sign of the last credited ACT `body_adv` (both magnitudes `> 1e-9`):

1. set a one-step conflict flag;
2. on the next motor loop, add frozen logit bias `+2.0` to HOLD and `-2.0` to ACT, then clear the flag;
3. apply a plastic `W_op` update `η|body_adv|(e_HOLD − e_ACT)`.

No capability-specific functions. Gate clear ≥13/16 floor-pairs and both population extras green. No DEVELOP before that.
