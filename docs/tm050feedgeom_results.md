# TM.0.50.FEEDGEOM results

## Decision: `states_separate_never_decode`

Product **0.0.004**. No v41 lock. No neural edit. Tick grid is diagnostic, not a fitted organism hyperparameter. TM049 first-match `feedback_not_action_separable` is unchanged.

Setup decoder cells passed 4/4 and are excluded from behavioral first-match. Transduction is adequate. Tanh is not saturating. Extra ticks do not become 4/4. Neutral does not rescue decoding. After one existing `_sensory_tick` of `motor_vec`, action states separate above the frozen 0.05 floor — and the existing motor decoder still ranks one default action.

DEV ran once on clean `72a9a4ae9d3bc7a343d87dc59064f0ab8a87012e`. Frozen runner SHA `504925a78645f32576e90e5b734a99dc31171471ae1a5db599a7be53b7452ba1`. 6 cells (2 setup + 4 scored). TM049/TM048/TM047/TM046 locks were not edited.

## Pattern flags

| Flag | Held |
| --- | --- |
| `collapse_before_recurrence` | false |
| `input_separates_tanh_compresses` | false |
| `later_ticks_decode` | false |
| `neutral_passes_cue_fails` | false |
| `states_separate_never_decode` | **true** |

## Cue `t=1` (the frozen TM049 transition, isolated)

`motor_vec` and `_x_tick` pairwise L2 ≈ 1.44. `W_in x` pairwise L2 ≈ 0.70. Post-tanh `ρ_1` pairwise L2 ≈ 0.59 (w0) / 0.58 (w1). Unit P1 pairwise L2 ≈ 0.17. All of those exceed 0.05. Fraction of units with \|tanh(pre)\| > 0.99 is **0**. Motor decode remains **1/4**, always the TM048 default (`h_901663069` on w0, `h_394767965` on w1).

At `t=0`, `ρ` is a single shared start state (L2 0), as required.

## Extra ticks and reset context

From `t=2` through `t=16`, `ρ` L2 plateaus near 0.62. Decode stays 1/4. Neutral/reset `ρ` traces the same curve: separable after one tick, never 4/4. Cue-state interference is not the failure.

## How this sits next to TM049

TM049's credit `observe()` wraps the same motor tick in start / cue symbols / `v_end` / `s_t`. That wrapped ρ was unique by hash but only 0.015–0.020 L2 — below the frozen floor — and unread by the decoder. Isolated at the insertion point, one `_sensory_tick(motor_vec)` already clears the floor. The missing usability is not transduction, saturation, tick count, or cue interference. Action experience and the motor decoder do not share a readable manifold.

This does not earn a new decoder, a stronger feedback rule, or a fitted tick count. Product remains 0.0.004.
