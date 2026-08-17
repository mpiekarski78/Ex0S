# CORTEX D7.R1 architecture amendment

Authorized only by [`cortex_diagnosis.fulldev_r5.lock`](cortex_diagnosis.fulldev_r5.lock).

Keep v16–v23 HOLD, habituation, S-write, RETRIEVE recency, echoic emit, and inter-observe vocal refractory. Add intra-utterance continuation: after the first EMIT inside one motor loop, bias subsequent inner ticks toward EMIT until STOP. Do not change `CONFLICT_HOLD_BIAS` or `VOCAL_REFRACTORY`. Do not edit `cortex_develop_scorers.py`.
