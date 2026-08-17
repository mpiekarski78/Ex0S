# CORTEX D5.R1 architecture amendment

Authorized only by [`cortex_diagnosis.fulldev_r3.lock`](cortex_diagnosis.fulldev_r3.lock).

Replace lifetime-max single-symbol novelty with habituation. Keep a decaying per-symbol familiarity trace. HOLD a lone symbol when its trace is below an absolute familiarization criterion (`FAMILIARITY_ABS`). Do not rank against other symbols' counts: ratio-to-max lets a repeated unknown become the max and stop HOLDing, which fails `unknown_holds>=12`.

Keep v16 equal-evidence / conflict / 3-symbol laws, v17 S-write recency, and v18 RETRIEVE recency tie-break. Do not edit `cortex_develop_scorers.py`.
