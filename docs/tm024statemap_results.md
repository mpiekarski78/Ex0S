# TM.0.24.STATEMAP results

**10 / 13** cells. Primary diagnosis: **`cue_collision_insufficient_separation`**.

Live candidate remains v29. No neural edit. n stays 64. Product **0.0.004**. Amendment not authorized.

| Cell | Result | What it showed |
| --- | --- | --- |
| S0 | PASS | One beneficial ACT moves identical-state `P(ACT)` 0.37→0.40 and beneficial handle score 0→1. |
| S1 | PASS | Same-cue `EVENT_END`→`EVENT_START` keeps the change. Cosine(teach, event) ≈ 1. |
| S2 | PASS | Delays 1/2/4/8 neutral ticks keep the change. |
| S3 | PASS | 1/2/4/8 irrelevant events keep the change. Cosine stays ≈ 1. |
| S4 | PASS | After `reset_ρ`, the cue reconstructs the same policy. Cosine ≈ 1. |
| S5 | PASS | A different preceding history does not remove the cue’s motor ranking. |
| S6 | **FAIL** | Two cues do not keep opposite actuator consequences. After the second teach, live cue A and live cue B read out the same policy. |
| S7 | PASS | Teaching vs probe cosine ≈ 1. Substitute and projection both succeed. **Aligned**, not a teacher/probe mismatch. |
| S8 | PASS | Motor ranking and `P(ACT)` both move. Not HOLD competition. |
| S9 | PASS | Fast, slow, and post-REST policies all keep the beneficial ranking. |
| S10 | **FAIL** | Age stages do not change `η_act` / `η_pred` / `β`. Live factory has empty `lineage_params`. |
| S11 | **FAIL** | 105 / 134 Arm D genes have a runtime effect. **29** are dead (region biases, neuromod, explore/eligibility/write thresholds, `age.*.explore_T`, `age.*.wm_persist`). |
| S12 | PASS | Independent renaming reproduces S0 pass and S7 `aligned`. |

## Decision table

The original hypothesis — teaching-state vs later autonomous probe-state misalignment — is **not confirmed**. S7 substitute, projection, and live probe all succeed, and the cosine is ~1.

S0–S5 pass because later ρ is essentially the teaching ρ. Distractors and event boundaries barely move the 64-unit state. That is why transfer looks easy for one cue, and why S6 cannot give two cues opposite consequences: the second credit writes through a nearly shared state, and both later probes read the same ranking.

S10/S11 are real but later on the ladder: the live organism does not load Arm D age rows, the default schedule is flat on learning rates, and 29 declared genes have no runtime reader.

Two-timescale state inside n=64 remains a **hypothesis**, not an authorized amendment. Increasing n is refused. Another lineage run stays closed. QUAL/EVAL sealed. Not 0.0.005.
