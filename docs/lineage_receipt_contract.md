# TM.0.59.RECEIPT contract

**Lab:** TM.0.59.RECEIPT · **Not a v41 candidate.** Product **0.0.004**. Diagnostic only. No neural or opaque-memory edit. Do not retune `EPISODE_MATCH_L2` (0.05). Do not rerun or edit TM058.

TM058 first-match `attempted_not_resident` stays frozen. Interpretation: `invalidated_measurement__observer_used_runner_provenance`. Architectural conclusion: none. The implementation followed the stronger law. The observer assumed runner-supplied identity would become organism identity.

## Observer

Call `write_opaque_kv`. Read the organism-assigned `provenance_id` from its write receipt. Locate the resident using that receipt identity. Compare resident key/value hashes with the attempted arrays. Repeat after checkpoint and capacity eviction.

The runner must never predict, provide, or reconstruct organism ids. It may only follow the receipt returned by the organism. The `write_opaque_kv(..., provenance_id=)` keyword is unused and is not identity.

## Question

Does receipt-following observation show opaque storage integrity: no merges, exact resident values, atomic eviction, byte-identical checkpoint?

## Probes

write (receipt identity plus no-merge);
capacity overflow (follow eviction and append receipts);
checkpoint restoration (locate by receipt ids).

## Pass

every accepted resident is found at the receipt `provenance_id`;
resident key/value hashes equal the attempted arrays;
capacity loss is the receipt's `evicted_provenance_id`;
checkpoint restore is byte-identical at those receipt ids.

## Ladder (setup excluded)

`setup_precondition_fail` → `cross_action_refresh` → `cross_cue_merge` → `observer_used_runner_provenance` → `resident_missing_at_receipt` → `attempted_hash_mismatch` → `unreported_capacity_loss` → `checkpoint_not_byte_identical` → `opaque_storage_integrity_holds`

## Refuse

Retune 0.05, neural edit, edit TM058, predict organism ids, skip runner-compat before DEV, v41, product-earn.
