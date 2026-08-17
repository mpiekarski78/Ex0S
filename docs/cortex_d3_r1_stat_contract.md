# CORTEX isolated D3.R1 statistical contract

**Lab:** TM.0.23.CORTEX.D3.R1.STAT  
**Product:** 0.0.004 · `earned_next=false` · `ex0s=null`  
**Authorized by:** [`cortex_diagnosis.fulldev_r1.lock`](cortex_diagnosis.fulldev_r1.lock)

Narrow D3 only. Reuse historical `score_d3` floors: `equal_holds>=8`, `clear_nonhold>=5`, clear more actionable than distractor. Do not edit `cortex_develop_scorers.py`. Do not lower `equal_holds`.

Worlds from a new sealed commitment, domain `TM023.D3.R1.`.

Authorized neural law (implement only after this apparatus is on `origin/main`):

1. **Equal-evidence HOLD:** if an observe presents two or more symbols and the last credited ACT had `|body_adv| ≤ 1e-9`, apply the existing one-step HOLD bias (`+2.0` HOLD / `−2.0` ACT) and the existing plastic HOLD-vs-ACT update.
2. **Low-familiarity HOLD:** if an observe presents exactly one symbol whose observation count is `< 0.5` times the most-observed symbol count (frozen ratio), apply the same HOLD response.

No capability-specific functions. Gate clear ≥13/16 pairs (main and twin D0+D3). No full D0–D12 and no nursery before that.
