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

The two eight-cue acquire fails (both `reg1`, four probes) are **case 3: value/consolidation**. Each live query’s unique nearest stored `key_rho` is the teach-index write (`slot == teach_index`, `d1 < d2`, 8/8 distinct slots). That row’s stored P1 then ranks the other actuator (`want != winner`, acquire `n_violations=3`, awake budget exhausted). Not case 1 (another key nearer) and not case 2 (missing/replaced row: append-only 1..8). Same-seed eight-cue **stable** after REST is 8/8 with zero store violations. Do not modify `R`. [`lineage_halfspace.failclass.lock`](lineage_halfspace.failclass.lock).

Frozen-runner note: `treatment_stable` also counts `kind=stable` scale cells. Core c2/c4/c8 stable and c8 hist all passed; one 8×4 scale-stable cell failed. That mix does not change this DEV’s first-match.

`separated_key` is a matched retrieval-path control, not a v36 train reproduction. Failed v36 control remains historical TM029.

