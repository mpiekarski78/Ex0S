# TM.0.58.STOREINT contract

**Lab:** TM.0.58.STOREINT · **Not a v41 candidate.** Product **0.0.004**. Diagnostic only. Do not implement the opaque-store write path in this freeze. Do not retune `EPISODE_MATCH_L2` (0.05). Do not change historical `_episode_write` while the flag is off. Do not resume drift or grounding.

TM057 first-match `storage_integrity_failure` stays frozen. It is a valid architectural falsification: a value payload cannot also define record identity.

## Law

When `opaque_store_enabled` is on, each accepted event writes an immutable `{provenance_id, key, value, when}`. Every accepted event is a distinct record. Value/P1 distance is diagnostic only. Capacity replacement is explicit generic eviction then append. Non-evicted residents remain byte-identical to the attempted \(v_t\).

The flag defaults off, survives checkpoints, and is not an `ACT_RECALL_MODE` or genome field.

## Question

Does the earned opaque-store law preserve storage identity?

## Probes

near values with different actions;
same action with different cue keys;
identical key/value repeated;
capacity overflow;
checkpoint restoration;
immutable provenance and exact resident hashes.

## Pass

zero cross-action refreshes;
zero cross-cue merges;
every non-evicted accepted value equals its attempted \(v_t\);
all loss is explicitly reported as capacity eviction.

## Ladder (setup excluded)

`setup_precondition_fail` → `cross_action_refresh` → `cross_cue_merge` → `attempted_not_resident` → `unreported_capacity_loss` → `checkpoint_not_byte_identical` → `storage_integrity_holds`

## Refuse

Retune 0.05, implement in this freeze, run DEV before the write path exists, resume TM057-style drift/grounding, v41, product-earn.
