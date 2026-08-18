# TM.0.58.STOREINT results

## Decision: `attempted_not_resident`

Product **0.0.004**. Canonical law unchanged. Floor **0.05** was not retuned. No v41 lock.

DEV ran once on clean `0db8bfd24906d6eb10e398d8991a3af01b36a4be`. Frozen runner SHA `2f7e497c216fe4a2fcdd5bbff73ed7f2fb7bd3f43c23a00630b4055ac278bb2d`.

The flag exists, defaults off, is not a genome or recall field, and survives checkpoint. `write_opaque_kv` does not call `_episode_write`.

## What the probes measured

Every scored cell had zero cross-action refreshes, zero cross-cue merges, and zero unreported loss. Capacity kept eight residents with one explicit eviction. Checkpoint restoration was byte-identical, including the flag.

The first-match is `attempted_not_resident` because the frozen observer looks up the `write_opaque_kv(..., provenance_id=)` keyword in resident rows. The eviction addendum assigns stored `provenance_id` from the organism counter and ignores that keyword. Residents are present; their IDs are `"1"`, `"2"`, … rather than `na0_0` / `ck0_0`.

Unit tests already show non-evicted residents equal the attempted arrays, copies isolate later mutation, invalid writes leave the store unchanged, and flag-off `_episode_write` is untouched.

## What this is not

Not a retune of 0.05, not a rewrite of the frozen runner, not a claim that FIFO is intelligent forgetting, not a v41 candidate, not a license to resume TM057 drift until this first-match is closed.
