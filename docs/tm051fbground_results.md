# TM.0.51.FBGROUND results

## Decision: `heldout_feedback_decode_fail`

Product **0.0.004**. No v41 lock. Existing decoder only. Wrapped TM049 states, not TM050 isolated insertion. TM050 first-match `states_separate_never_decode` is unchanged.

Setup `reference_only` decoder cells passed 4/4 and are excluded from behavioral first-match. Held-out wrap decode on the grounded checkpoint remains **1/4**. The existing `W_act_query` did move under generic development with the feedback flag on — ordinary reference P1s dropped from 4/4 to 2/4 and 1/4 — but that update did not produce a readable held-out wrap mapping. Shuffled grounding also stayed 1/4 on true labels, so the shuffle control is intact. No-memory is not reached.

DEV ran once on clean `10ec197da1efa85c2f4884a5076786fc07742f4b`. Frozen runner SHA `f73c8671db2f3bac6f6b4e22eb08687d933559e476d5c2adb2ff4c879c230706`. 10 cells (2 setup + 8 scored). TM050/TM049/TM048/TM047/TM046 locks were not edited.

## Causal chain

Required: ρ_feedback → 4/4 motor decode → S value → 4/4 reinstatement → 4/4 canonical, while no-memory fails.

Stopped at held-out wrap decode. Four unique wrapped P1 hashes per cell, same 1/4 collapse as TM049.

## Cells

| Cell | Code | Wrap true | Notes |
| --- | --- | --- | --- |
| `decoder\|w0` | `decoder_ok` | — | 4/4 ordinary development P1s |
| `decoder\|w1` | `decoder_ok` | — | 4/4 ordinary development P1s |
| `reference_only\|w0` | `reference_ok` | 1/4 | TM049 default `h_901663069` |
| `reference_only\|w1` | `reference_ok` | 1/4 | TM049 default `h_394767965` |
| `feedback_grounded\|w0` | `heldout_feedback_decode_fail` | 1/4 | still `h_901663069`; ordinary 2/4 |
| `feedback_grounded\|w1` | `heldout_feedback_decode_fail` | 1/4 | collapsed to `h_335854708`; ordinary 1/4 |
| `shuffled_grounding\|w0` | `shuffled_ok` | 1/4 | true labels not 4/4 |
| `shuffled_grounding\|w1` | `shuffled_ok` | 1/4 | true labels not 4/4 |
| `feedback_no_memory\|w0` | `heldout_feedback_decode_fail` | 1/4 | same grounded checkpoint |
| `feedback_no_memory\|w1` | `heldout_feedback_decode_fail` | 1/4 | same grounded checkpoint |

## What this is not

This is not a memory-addressing failure, a signal-strength failure, or a new-decoder earn. Reference wrap is still the TM049 1/4 control. Shared-decoder interference is not first-match: wrap never reached 4/4. Product remains 0.0.004.
