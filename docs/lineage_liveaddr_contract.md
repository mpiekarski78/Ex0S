# TM.0.41.LIVEADDR contract

**Lab:** TM.0.41.LIVEADDR · **Not a v40 candidate.**

Locked TM040 telemetry cannot name the scoring address: `probe_map` used `actuator_scores(live P1)` and stored no path/slot/familiarity/d1/R/hashes. This lab reconstructs the four failed TM040 acquire stems (`TM040.CAUSAL.DEV.`, seed `404000039`, w0/w1, both orders) and classifies the failed live cue **before** any architecture change.

Do **not** edit `neural_cortex.py` or `joint_socp.py`. Do **not** invoke SOCP unconditionally, add live constraints, change the fallback trigger, or modify half-spacing. Do **not** rewrite TM040 historical locks.

## Measurements (same scorer: `actuator_scores`)

For each cue, under `v37` / `fallback_joint` / `always_joint` weights:

- canonical `actuator_decision_scores`: path, familiar, d1, d2, R, slot, scoring-address hash
- expected teach-index slot vs retrieved slot
- store ranking/γ/violation on the expected stored P1
- TM040-style live ranking/γ on live P1
- counterfactual expected stored P1 through `actuator_scores`
- counterfactual live scoring address through `actuator_scores`

## Failed-cue classes (first-match)

1. `cortical_fallback_unfamiliar` — half-spacing/reinstatement; store-only SOCP cannot see the live address
2. `wrong_slot` — episodic completion to the wrong row
3. `canonical_path_inconsistency` — correct slot and stored P1 pass; TM040 live P1 still fails (probe vs organism path)
4. `store_pass_live_addr_fail` — stored P1 passes; the live scoring address fails (consolidation does not cover the live query)
5. `stored_p1_fails` — forced stored P1 fails despite zero store violations (measurement bug)

`always_joint` is diagnostic: if retrieval hashes/path/slot match fallback while live ranking flips, SOCP moved \(W\) rather than retrieval (expected: SOCP cannot change keys).

Later-learning remains **not_exercised**. Contradict remains jointly-feasible atomic apply. Product **0.0.004**.
