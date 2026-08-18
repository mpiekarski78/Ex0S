# TM.0.54.PROV results

## Decision: `state_generator_mismatch`

Product **0.0.004**. No v41 lock. SOCP was not scored. Floor **0.05** was not retuned.
TM053 first-match remains `coverage_generalizes`. Interpretation remains `invalidated_measurement__duplicate_state_support`. Architectural conclusion remains **none**.

Value need not carry cue context. `k` is cue/context identity; `v` is action experience.

DEV ran once on clean `4e7706ca8e5e6dcd739e368ad6a0538030cba2e4`. Frozen runner SHA `cbfe1c37de4532141d2e770ae03b2d50a32f720276fe6f2064c9a7f7b4ee685d`.

## Gate

Pinned TM052 reconstruction succeeded: write-time last-P1 hashes, held-out wrap hashes, and parent `W_act_query` all matched.

Canonical N=1 **held-out** hashes match the pins. Canonical N=1 **training** hashes do not.

The first divergence is `frozen_dev_probe_vs_write_last_p1` on both worlds: a frozen observe-clamp-observe on the development cues is not the same state as the P1 written during plastic grounding.

| Check | w0 | w1 |
| --- | --- | --- |
| Native write last-P1 vs pin | match | match |
| Native hold wrap vs pin | match | match |
| Frozen hold probe vs pin | match | match |
| Frozen s_dev probe vs write last-P1 | **mismatch** | **mismatch** |
| Reset changes hold | no | no |
| Frozen two-cue wraps action-invariant | yes | yes |

On the frozen TM052 parent, extra cues produce the same wrap as `s_hold` (action-locked `v`). That is not a defect in the value code. It also means TM053’s collapsed curve was a frozen-probe family, not TM052’s write-time training states.

ρ after `v_end` is not ρ after `s_t`. Wrapped P1 is the post-`v_end` snapshot; `rho_feedback` is after the `s_t` tick. Pending is consumed. `same_ix` is 0 on these tags.

## Fork (not taken)

Hashes did not both match, so this wall does not earn one-exemplar grounding or a genuine coverage curve. The remaining question is which of the two organism procedures is the credit-and-write lifecycle for grounding:

1. last P1 written during plastic development
2. frozen observe-clamp-observe wrap after checkpoint

Do not redesign representation until that is chosen. Do not force context into `v`.

## What this is not

Not a coverage result, not an installed oracle, not a fitted N, not a second decoder, not a v41 candidate.
