# TM.0.46.ONESHOT results

## Decision: `generic_reinstatement_fail`

Necessary cell: none. Earned K/Q/V: **false**. Earned interface: **true**.

Product **0.0.004**. No `cortex.candidate.v41.lock`. K/Q/V not tuned.

The motor decoder expressed every action from valid reinstated **development** states (4/4 handles, both worlds). Slow `W_act_query` then stayed frozen while one-shot test facts were written. The symbolic oracle **completed episodic reinstatement** (`memory_path=episodic_completed`) but ranked 1/4 facts on immediate probe. `no_persistent_memory` failed the same cells from live ρ. That is a generic reinstatement-to-motor miss, not a clean S-necessity gap.

DEV ran once on clean `ee1102d4e36c00adc69220d61d17e3493673d9dc`. Frozen runner SHA `8dbadd143f0fed629496a70c9d6288e60c65301fadd392cab6e3d77ea0b5d6b0`. 34 cells. TM044/TM045 locks were not edited.

## Cell brief

- Decoder precondition: pass on w0 and w1.
- Frozen oracle / opaque / no-memory: 1/4 on immediate, delayed, and distractor (both worlds). Oracle telemetry shows completed reinstatement, then wrong cortical ranking.
- Observational `slow_cortex_enabled`: immediate w0 4/4 (consolidation can absorb these facts); other slow cells mixed. Not a gate.

Do not repair learned addressing on this wall. The earned site is the generic reinstatement-to-motor interface.
