# TM.0.40.CAUSALBATTERY.R2 contract

**Lab:** TM.0.40.CAUSALBATTERY.R2 · **Not a v40 candidate.**

TM040's recorded first-match stays on file. Scientifically it is not a valid organism acquire failure: the acquire gate scored ablated raw P1. This R2 is a harness-boundary repair on **fresh** domains. TM041 reconstructions are diagnostic evidence, not a replacement score.

## Canonical probe

Motor ACT and experiments use one API:

`actuator_decision_scores(live_p1) → scores, scoring_address, recall_meta`

via `experiments/canonical_act_probe.py`. Raw `actuator_scores(live_p1)` may exist only as `diagnostic_raw_live_scores`. Never a behavioral gate.

Perturbation modifies **live P1**, then reruns retrieval and canonical scoring.

## Unchanged

No neural, SOCP, recall-mode, fallback-trigger, half-spacing, or threshold edits. Store predicate remains `_episode_rehearsal_violation` on stored P1.

## Battery

Same 102-cell causal battery as TM040 (`fallback_joint` scored; `always_joint` observational; `v37` control) on unused `TM040.CAUSAL.R2.*` worlds.

Every behavioral probe records live-P1 hash, scoring-address hash, retrieval path, and retrieved slot.

## Refuse

Rewrite TM040 DEV/decision; candidate lock; treat always-joint as having behaviorally fixed TM040 acquire.
