# CORTEX isolated D3.R2 statistical contract

**Lab:** TM.0.23.CORTEX.D3.R2.STAT  
**Product:** 0.0.004 · `earned_next=false` · `ex0s=null`  
**Authorized by:** [`cortex_diagnosis.v14.lock`](cortex_diagnosis.v14.lock)

Narrow D3 only. Historical `score_d3` floors unchanged. Do not edit `cortex_develop_scorers.py`. Do not lower `equal_holds`. Do not drop C6.

Worlds from a new sealed commitment, domain `TM023.D3.R2.`.

Authorized neural law (implement only after this apparatus is on `origin/main`):

1. **Equal-evidence HOLD:** if an observe presents two or more symbols, the last credited ACT had `|body_adv| ≤ 1e-9`, **and** at least **3** distinct symbols have been observed in this life, apply the existing one-step HOLD bias (`+2.0` HOLD / `−2.0` ACT).
2. **Low-familiarity HOLD:** if an observe presents exactly one symbol whose observation count is `< 0.5` times the most-observed symbol count, apply the same HOLD response.

Frozen `EQUAL_EVIDENCE_MIN_SYMBOLS = 3`. This keeps C6 (only `a`,`b`) from triggering equal-evidence HOLD. No capability-specific functions. Gate clear ≥13/16 pairs (main and twin D0+D3). Re-earn C4/C5/C6 before reveal.
