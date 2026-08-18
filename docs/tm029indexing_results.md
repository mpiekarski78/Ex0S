# TM.0.29.INDEXING DEV

Decision: **indexing_core_stability_fail**.

v36 hippocampal indexing on fresh `TM029.INDEXING.*` worlds. Write/replacement remained P1 L2 0.05. Recall used action-owned sparse 8-of-64 keys with 5/8 familiarity. Product **0.0.004**.

## Operational

Treatment **8/8 acquire** at 2/4/8 cues, including eight-cue ranking. Core **c2/c4/c8 stable** ranking+geometric+perturbation all pass. Twin, eco, spec, integrity, and 8×4 scale acquire pass.

First-match **indexing_core_stability_fail** because hist failed the perturbation gate (4/4 rank and γ, 0/4 pert). Core taught `stable|c2/c4/c8` is 12/12. The frozen `treatment_stable` flag also mixes 8×4 scale-stable cells (same TM028-style kind==stable); those likewise fail pert only. Even excluding scale, hist alone makes the first-match valid. Novel cells 0/4: sparse overlap was **6/8** (above the 5/8 null gate) with `path=episodic_completed` — correlated organism keys, not HOLD. Same frozen DEV execution refused; runner SHA not edited.

## Causal observations (not architectural failures)

- `off` ablation stable fails (`reinstatement_wall`) — episodic recall was not unnecessary.
- `raw_p1` ablation stable fails (`reinstatement_wall`) — v35 nearest-P1 failure reproduced on these worlds.
- `early_raw` ablation taught-stable **passes** (4/4) — earlier capture without sparse hashing already supports taught probes; sparse hashing is **not** isolated on this contrast.
- `separated_key_no_familiarity` does not reject novel cues; treatment also does not — familiarity was not shown causal here.

No `cortex.candidate.v36.lock`. Same frozen DEV execution refused.
