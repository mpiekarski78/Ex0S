# EX0S-DEV1 Stage B Contract

This document is intentionally **non-normative**.

It is a stage contract and planning note for Stage B. It is **not** a
preregistration lock, and it does **not** authorize execution. The real
`docs/exos_dev1.stage_b.prereg.lock` must be created only after Stage A
passes an untouched confirmation tranche and Stage B is actually ready to run.

## Unlock condition

Stage B remains locked until:

- Stage A passes untouched confirmation
- a Stage A winner is frozen
- fresh Stage B world seeds are allocated
- the executable Stage B runner and interventions are fixed

## Stage B claim

Stage B tests fast-memory ownership.

Slow cortical plasticity and consolidation must be disabled during one-shot
fact acquisition. Probes occur before any cortical transfer. This prevents
slow cortex from solving the one-shot probe and falsely reproducing
`memory_not_necessary`.

## Required gates

- No-H arm fails one-shot probes
- H arm passes one-shot probes, with slow cortex disabled in both arms
- one-shot facts survive `EpisodeReset`
- early H wipe removes the facts
- `FullCheckpoint` restore works without harness re-keying
- `HippocampalGraft` between matched donor twins redirects the facts
- renamed cues and new worlds pass
- the organism generates every retrieval address

## Matched donor twin definition

Matched donor twins are clones of the same pre-teaching `FullCheckpoint`,
followed by different fact experiences before the graft. An unrelated newborn
is not expected to decode another organism's private neural coordinates.

## Planned intervention constraints

- `h_write_disabled = false`
- `h_read_disabled = false`
- `slow_cortex_consolidation_disabled = true`
- `consolidation_disabled_for_no_H_arm = true`
- `consolidation_disabled_for_H_arm = true`
- probes must occur before any cortical transfer

## Execution-time fields still to freeze

These belong in the future Stage B preregistration lock, not here:

- `implementation_sha`
- `runner_schema_sha`
- `genome_hash` of the Stage A winner
- Stage B world seeds
- budget, thresholds, and untouched confirmation tranche
- backend / numeric mode actually used for scored execution
