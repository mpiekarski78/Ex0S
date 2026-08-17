# CORTEX isolated D5.R1 statistical contract

**Lab:** TM.0.23.CORTEX.D5.R1.STAT  
**Product:** 0.0.004 · `earned_next=false` · `ex0s=null`  
**Authorized by:** [`cortex_diagnosis.fulldev_r3.lock`](cortex_diagnosis.fulldev_r3.lock)

Narrow D5 only. Historical `score_d5` unchanged. Do not edit `cortex_develop_scorers.py`. Do not lower `unknown_holds>=12` or `known_nonhold_rate>=0.30`.

Worlds from a new sealed commitment, domain `TM023.D5.R1.`.

Each isolated life runs D0, binds actuators, runs the same light `[a,b]` teach as full-dev, then historical `score_d1`–`score_d4` so earlier symbols exist, then `score_d5`. Pair clear requires D0 and D5 only. D1–D4 are transfer diagnostics, not this gate's floor.

Authorized neural law (implement only after this apparatus is on `origin/main`):

1. Keep v16 HOLD laws except the lifetime-max single-symbol novelty test.
2. Keep v17 S-write recency and v18 RETRIEVE recency tie-break.
3. **Habituation familiarity:** maintain a decaying per-symbol familiarity trace. A lone symbol HOLDs only when that trace is below an absolute familiarization criterion. Do not compare to other symbols' lifetime or recency maximum — a repeated unknown must not become "familiar" merely by being the current max.

Gate clear ≥13/16 pairs (main and twin D0+D5). Re-earn C4/C5/C6 before reveal.
