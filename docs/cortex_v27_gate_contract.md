# CORTEX isolated v27 generality gate contract

**Lab:** TM.0.23.CORTEX.V27.GEN  
**Product:** 0.0.004 · `earned_next=false` · `ex0s=null`  
**Authorized by:** [`cortex_diagnosis.v26_generality.lock`](cortex_diagnosis.v26_generality.lock)

Fresh sealed commitment, domain `TM023.V27.GEN.`. Not `pair_seeds`. Not FULLDEV.R*. Not D3–D7.R*.

## Required

| Check | Pass |
|-------|------|
| G1 | `neural_cortex.py` and `cortex_memory.py` have zero hits on the GENERALITY.v26 forbidden identifiers/regexes |
| G3 | each life: after heard-A→emit-B teaching, 20 probes of A have `emit_B > echo_A` and `emit_B ≥ 3` |
| G5 | each pair: main taught STOP boundary 2, twin taught 4, same 4-token sensory length; `abs(mean4-mean2) ≥ 1.0` and each life closer to its boundary |
| C4/C5/C6 | re-earned on the v27 candidate before reveal |

Pair clear: both lives G3-green **and** the pair G5-green. Gate clear: G1 green **and** ≥13/16 pairs.

D0 is recorded. Isolated D6/D7 are diagnostic, not required. FULLDEV.R7 stays sealed.
