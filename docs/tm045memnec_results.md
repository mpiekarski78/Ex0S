# TM.0.45.MEMNEC results

## Decision: `memory_never_necessary`

Necessary cell: none.

Product **0.0.004**. No `cortex.candidate.v41.lock`. K/Q/V not tuned. Learned-arm accuracy is observational and is not a gate.

First-match on the frozen ladder: 2-cue immediate oracle passed both worlds, then no frozen cell had oracle success with `no_persistent_memory` failure. Whenever the no-memory arm failed (8 cues), the oracle failed the same cells. Persistent information was therefore not required on this wall.

DEV ran once on clean `0cb92cca236e3a066e067c6f3d548c41e081ac7a`. Frozen runner SHA `10717d65aa7d851a5fe4c880413a3d0b0fc23695dfba38083de6fa3466566212`. 72 cells. SOCP stayed off. `joint_socp.py` and TM044 runner/DEV/decision were not edited.

## Cell brief

- 2-cue and 4-cue: oracle pass, no-memory pass, all four conditions, both worlds.
- 8-cue immediate/delayed/distractor: oracle 7/8 and no-memory 7/8 on both worlds (same miss; not a necessity gap).
- 8-cue revision: oracle fail and no-memory fail on both worlds.
- Learned projection: fail on 2-cue and 4-cue association-like cells; opaque reinstatement flipped already-correct live cortical scores. Observational only.

No earned site for projection-learning repair. This is not a v41 candidate review.
