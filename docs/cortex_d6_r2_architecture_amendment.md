# CORTEX D6.R2 architecture amendment

Authorized only by [`cortex_diagnosis.v21.lock`](cortex_diagnosis.v21.lock).

Keep v21 echoic emit. After EMIT, briefly bias HOLD. After HOLD, if echoic is nonempty, briefly bias EMIT. Do not change `CONFLICT_HOLD_BIAS`. Do not edit `cortex_develop_scorers.py`.
