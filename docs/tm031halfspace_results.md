# TM.0.31.HALFSPACE results

v37 early-raw half-minimum-spacing familiarity. Product **0.0.004**.

## Decision: `halfspace_core_acquire_fail`

Geometry-derived conservative familiarity rule. Not a TM030 numeric cutoff. Not a novelty theorem.

- treatment acquire 2/4/8: True / True / False
- treatment stable+hist: False
- novel pass: True
- matched gate path toggle: True
- half_spacing_gate_causal: False

Eight-cue treatment acquire is 22/24. First-match stops there, so the battery does not claim the gate as causal even though novelty cells rejected (`familiar=false`, `path=cortical_fallback`) and matched ON/OFF queries toggled path. `early_raw` taught-stable clones passed as an observation.

The two eight-cue acquire fails (both `reg1`) retrieved a unique stored episode **inside** the familiarity ball (`path=episodic_completed`, `d1 < R`) but the wrong handle. That is identity error under a conservative ball, not novelty-style fallback.

Frozen-runner note: `treatment_stable` also counts `kind=stable` scale cells. Core c2/c4/c8 stable and c8 hist all passed; one 8×4 scale-stable cell failed. That mix does not change this DEV’s first-match.

`separated_key` is a matched retrieval-path control, not a v36 train reproduction. Failed v36 control remains historical TM029.

