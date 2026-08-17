# CORTEX D7.R2 architecture amendment

Authorized only by [`cortex_diagnosis.v24.lock`](cortex_diagnosis.v24.lock).

Keep v16–v23 HOLD, habituation, S-write, RETRIEVE recency, echoic emit, and inter-observe vocal refractory. Replace softmax `UTTERANCE_PERSIST` with a motor-program unroll: after the first EMIT, continue the just-heard phrase, optionally reduplicate a short chunk, then STOP. Do not change `CONFLICT_HOLD_BIAS` or `VOCAL_REFRACTORY`. Do not edit `cortex_develop_scorers.py`.
