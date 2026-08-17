# CORTEX D5.R3 architecture amendment

Authorized only by [`cortex_diagnosis.fulldev_r6.lock`](cortex_diagnosis.fulldev_r6.lock).

Keep v16–v25 HOLD, habituation, S-write, RETRIEVE recency, echoic emit, vocal refractory, and familiar phrase-final STOP. Block motor-program unroll when any current-phrase symbol has familiarity below `FAMILIARITY_ABS`; HOLD instead. Do not change `CONFLICT_HOLD_BIAS`, `VOCAL_REFRACTORY`, or `FAMILIARITY_ABS`. Do not edit `cortex_develop_scorers.py`.
