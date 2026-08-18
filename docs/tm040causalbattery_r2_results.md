# TM.0.40.CAUSALBATTERY.R2 results

Canonical-probe causal battery on fresh `TM040.CAUSAL.R2.*` worlds. Product **0.0.004**. No `cortex.candidate.v40.lock`.

## Historical TM040 (immutable)

- recorded first-match: `jointsocp_fallback_acquire_fail`
- interpretation: `invalidated_measurement__canonical_path_mismatch`
- architectural conclusion: **none**

TM041 reconstructions were diagnostic, not a replacement score. always-joint did not behaviorally fix TM040 acquire; it ranked the ablated raw-P1 address.

## R2 decision: `canonical_r2_later_learning_not_exercised`

Fallback_joint **passes** acquire, stability, history, novelty, reversal, specificity, 8×4, later-learning probes, and contradict. Untouched eight-cue acquire is live **8/8** with zero store violations under the canonical motor path (`actuator_decision_scores`), including v37. Fallback still does not invoke SOCP on those acquire cells.

SOCP did activate on four untouched cells (scale). Later-learning after an installed SOCP was **not exercised** (`n_later_after_socp=0`). That is the first-match. Still not a candidate lock.

always_joint observational read: `scheduling_efficiency`. That is not a claim that always-joint repaired organism acquire.

## Measurement

Behavioral probes used `experiments/canonical_act_probe.py`. Every probe recorded live-P1 hash, scoring-address hash, retrieval path, and retrieved slot. Raw live scores were diagnostic only.
