# TM.0.60.ONDRIFT contract

**Lab:** TM.0.60.ONDRIFT · **Not a v41 candidate.** Product **0.0.004**. Diagnostic only. No neural edit. No SOCP installation. Do not install W\*. Do not retune `EPISODE_MATCH_L2` (0.05). Do not rerun or edit TM055–TM059.

TM055 first-match `setup_precondition_fail` stays frozen. TM057 first-match `storage_integrity_failure` stays frozen. TM058 first-match `attempted_not_resident` stays frozen as an invalidated measurement. TM059 first-match `opaque_storage_integrity_holds` stays frozen. Receipt identity is the observer law.

## Distinction

- **Attempted write-time \(v_t\)**: the event-time value offered to `write_opaque_kv`. Offline diagnostic ceiling for representational stability, including later-evicted events.
- **Organism-available residents**: currently resident values located only by the organism write receipt `provenance_id`. What consolidation could actually learn from S.
- **Held-out**: genuinely later write-time values. Not reconstructed wraps. Not runner-predicted `"1"`, `"2"`, …

## Method

One continuous action-balanced trajectory with unique cues. Enable `opaque_store_enabled`. Call `write_opaque_kv`. Read `provenance_id` from the write receipt. Locate the resident with that receipt identity.

Train diagnostic \(W_N^*\) on a chronological prefix plus ordinary reference-action constraints:

1. prefix of all attempted write-time \(v_t\);
2. currently resident receipt-identified values after that prefix.

Test both on later write-time values. Discard every \(W^*\).

## Ladder (setup excluded)

`setup_precondition_fail` → `observer_used_runner_provenance` → `prefix_infeasible` → `representation_drift` → `capacity_eviction_limits_consolidation` → `generic_grounding_consolidation_earned`

| Result | Interpretation |
| --- | --- |
| Attempts fail future states | Representation drift |
| Attempts generalize, residents fail | Capacity/eviction limits consolidation |
| Both generalize | Generic grounding consolidation becomes an earned mechanism |

## Refuse

Neural edit, SOCP install, install W\*, predict organism ids, use the `write_opaque_kv` keyword as identity, retune 0.05, rewrite TM055–TM059, v41, product-earn, DEV before runner-compat against the law addendum.
