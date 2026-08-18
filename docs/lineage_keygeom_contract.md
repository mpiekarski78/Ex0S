# TM.0.30.KEYGEOM contract

**Lab:** TM.0.30.KEYGEOM · **Organism:** v36 (frozen; no neural edit)

Fresh `TM030.KEYGEOM.DEV./TWIN.` worlds. **8 checkpoint cells** (core c8 + 8×4 scale). Standalone runner: do **not** patch TM028, TM029, or `gr` module globals.

## Purpose

Frozen key-geometry diagnostic wall before v37. Measure within-cue, between-cue, and novel-key geometry to route among event-state isolation, learned pattern separation, or a different familiarity signal. **Not** a recall pass/fail battery and **not** a threshold-tuning pass.

## Side-effect-free probes

Every baseline, history, and perturbation probe runs on a **checkpoint clone** with plasticity frozen. Geometry extraction must not mutate the trained parent: no credit, episode writes, REST, counters, or shared RNG changes on the source checkpoint.

## Dual retrieval identity

Record **both** raw and separated retrieval on each probe:

| Path | Fields |
|------|--------|
| Raw (`key_rho`) | nearest stored slot, best/second L2 distance, margin |
| Sparse (Hadamard 8-of-64) | best/second overlap slot, overlap values, tie flag |

## Expected baseline slot

Each taught cue carries a stable **logical_id** (teach order) and an **expected_physical_slot** from the preregistered baseline probe on the frozen checkpoint. Identity drift compares retrieved slots to this baseline, not to mutable row order alone.

## Pinned probes

Perturbation vectors, history symbols, novel symbols, trial counts, domains, and manifest are pinned in prereg before execution. Same σ and trial count as TM029 margin table (`rho_perturb_sigma=0.01`, `perturb_n=20`).

## Distributions (not maxima only)

Report distributions across cells and cues:

- within-cue L2/cosine changes (baseline vs hist / pert)
- raw and sparse identity-drift rates
- taught-key off-diagonal sparse overlaps
- novel best/second overlaps and false-familiarity rate
- retrieval margins and tie rates

## Decision (descriptive only)

No post-DEV routing thresholds and no v37 architecture in this pass. Decision record exposes:

- `key_rho_history_drift`: observed / not_observed
- `key_rho_perturbation_drift`: observed / not_observed
- `separator_added_collisions`: observed / not_observed
- `novel_overlap_false_familiarity`: observed / not_observed
- `geometry_wall_complete`: true

Definitions are frozen in prereg `outcome_definitions`. Product **0.0.004**.

## Wording

v36 killed the **event-end P1 retrieval** story specifically (`raw_p1` reinstatement wall). It did **not** kill raw retrieval generally — `early_raw` passed ordinary taught-stable probes on TM029.
