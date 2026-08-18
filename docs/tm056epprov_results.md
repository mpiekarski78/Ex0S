# TM.0.56.EPPROV results

## Decision: `replaced_under_write_law`

Product **0.0.004**. Canonical law unchanged. Floor **0.05** was not retuned. No v41 lock. No SOCP.

TM055 first-match remains `setup_precondition_fail`. It is a valid setup stop.

DEV ran once on clean `7d62fe0d7c632b25340706904f5d99004ce15b55`. Frozen runner SHA `8e89a96393ed2543247d9de3f7a8a019073e62e948d8f94d09c66b09644be656`.

Targets are `_episode_write` identities. `episodes[-1]`, handle lookup, and live `_last_p1` were comparators only.

## What was recovered

Both worlds reproduced TM052’s parent `W_act_query` and TM052’s pinned hashes **when measured the TM051 way** (last valid episode per handle while scanning the store). REST wrote nothing. REST did not mutate those write-identity P1s. After `s_t` and after `ground_one`, stored P1 still equalled the value just written.

There are eight episode slots. Sixteen grounding events therefore cannot all remain distinct records.

| Cycle | What `_episode_write` did |
| --- | --- |
| 1 | four inserts |
| 2 | two in-place refreshes (`d ≤ 0.05`, new `v_t` not stored) and two inserts |
| 3 | two inserts that **are** the TM052 pins for two actions, then in-place / evict |
| 4 | three evicts (new `v_t` stored) and one in-place refresh (new `v_t` discarded) |

The last grounding `v_t` for one action in each world was absorbed under the existing 0.05 write law: the argument hash is not the stored P1. That is lifecycle/replacement, not representation drift.

The TM052 pins are surviving earlier writes still sitting in the eight-slot store. Handle-scan last-in-list is not write provenance. Live `_last_p1` at measurement is not the pin list.

## What this is not

Not a drift curve, not a license to retune 0.05, not a law change, not an installed oracle, not a v41 candidate.

The next drift curve must train and test on write-identity records — the value passed into `_episode_write` — not on a later handle scan.
