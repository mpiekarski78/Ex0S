# CORTEX v27 architecture amendment

Authorized only by [`cortex_diagnosis.v26_generality.lock`](cortex_diagnosis.v26_generality.lock).

Remove scripted current-observe phrase replay. Keep habituation, equal-evidence HOLD, S-write, RETRIEVE recency, and neophobia HOLD on an unfamiliar **single** symbol.

Authorized neural law (implement only after this apparatus is on `origin/main`):

1. Delete `_phrase`, `phrase_program`, `phrase_target`, phrase-final hard STOP, and length-2 reduplication.
2. Echoic persistence may bias EMIT token ranking. It must not hard-emit the heard queue as the answer.
3. Continuation/refractory applies after EMIT **or** ACT, not EMIT-only.
4. STOP is chosen from evidence, never assigned because a copied token queue emptied.
5. Do not add `known_chunks`, stage/domain branches, stored expected lengths, or capability-named shortcuts.
6. Do not edit `cortex_develop_scorers.py`. Do not lower D5/D6/D7 floors. Do not reveal FULLDEV.R7.

Narrow gate: G1 source-clean + G3 non-echo + G5 STOP evidence. C4/C5/C6 required after the edit. Isolated D6/D7 are diagnostic only.
