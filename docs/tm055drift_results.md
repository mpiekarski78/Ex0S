# TM.0.55.DRIFT results

## Decision: `setup_precondition_fail`

Product **0.0.004**. No v41 lock. Every diagnostic W\* was discarded. Floor **0.05** was not retuned.
Canonical generator remains write-time last P1. Frozen wraps were not used as canonical.

TM054 first-match remains `state_generator_mismatch`. TM053 first-match remains `coverage_generalizes`.

DEV ran once on clean `1a933f439f5f2e72bbbfaec50c781c253cfb683f`. Frozen runner SHA `23a3002029560e6a83d5ae5646e5631101fee6b89981e8cd814688e87f9a392b`.

## Gate

The TM051-shaped reconstruction reproduced pinned TM052 parent `W_act_query` on both worlds. It did not reproduce pinned TM052 training last-P1 hashes.

Those pins are last episode P1 after rest and eval prep. This wall captured `ep["p1"]` immediately after each grounding event. Same life, same W0, different value snapshots. That is why the coverage question is not scored.

`live_match` also failed: after `ground_one` returns, live `_last_p1` is often not the stored episode P1. That matches TM054’s split between the post-`v_end` write and the later `s_t` tick. The stored trace is the episode P1.

Ordinary reference-action constraints on the new TM055 lives were 4/4.

## What was not decided

Prefix SOCP cells ran, but they are not a first-match. No claim of consolidation, drift, clusters, or migration is earned until a reconstruction captures the same write-time snapshots TM052 actually recorded — or until a later wall drops the TM052 pin and treats a new event-time tape as its own universe.

Keys were unique per event (`n_unique_k=32`). Values were not (`n_unique_v=21`). That is observational only.

## What this is not

Not a coverage curve, not an installed oracle, not a frozen-wrap result, not a value-code redesign, not a v41 candidate.
