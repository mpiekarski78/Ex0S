# CORTEX v13 statistical motor-learning contract

**Lab:** TM.0.23.CORTEX.V13.STAT  
**Product:** 0.0.004 · `earned_next=false` · `ex0s=null`  
**Authorized by:** [`cortex_diagnosis.v12.lock`](cortex_diagnosis.v12.lock)

Retain v12 D1 press+harm floors, population extras, swapped D2 HOLD window, sealed-seed worlds, and git-pinned reveal. Keep `holds>=5`, `beneficial>=3`, `rho_ok`. Do not lower `holds>=5`. Do not raise `CONFLICT_HOLD_BIAS`.

Authorized neural law (implement only after this apparatus is on `origin/main`):

Maintain a scalar advantage baseline `ema` (birth `0`). When a credited ACT has `|body_adv| > 1e-9`:

1. if `|ema| > 1e-9` and `ema * body_adv < 0`, set the existing one-step conflict flag and apply the existing plastic `W_op` HOLD-vs-ACT update; **do not** update `ema`;
2. otherwise `ema ← (1 − α) ema + α body_adv` with frozen `α = 0.05`.

Conflict response remains v12: next motor loop `+2.0` HOLD / `−2.0` ACT, then clear the flag. No capability-specific functions. Gate clear ≥13/16 floor-pairs and both population extras green. No DEVELOP before that.
