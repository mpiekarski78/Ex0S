# TM.0.49.ACTFEED results

## Decision: `feedback_not_action_separable`

Product **0.0.004**. No v41 lock. K/Q/V not tuned. Decoder not retrained. Learned addressing not earned.

Setup decoder cells (`decoder|w0`, `decoder|w1`) passed 4/4 and are excluded from behavioral first-match. Six scored cells remain. Scalar-only still reproduces the TM048 default collapse. The frozen `_sensory_tick` channel does inject `motor_vec`, but the resulting action dependence is smaller than the episode-match floor.

DEV ran once on clean `30c69d8df99c2246dba5a597cae2356d5c8126e1`. Frozen runner SHA `3def01d5502b28a5ffafeab58b07ee481d5748e5c765b1cbbf52d1c1ed6f275d`. Neural SHA after the authorized edit `a33f04479716d21624f9f8d0167ceaf4a658fd57a9070b058933f71fa1ae155c`. 8 cells (2 setup + 6 scored). TM046/TM047/TM048 locks were not edited.

## What the channel did

Transition was exactly the freeze: one `_sensory_tick(pending.motor_vec)` through existing `_x_tick` → `W_in`, tanh, `record_sensory=True`. Storage used consumed cue `pending.key_rho` and post-feedback `_last_p1`. Teacher clamp stayed on public `clamp_action`.

On `action_feedback`, four actions produced **four unique** feedback-ρ hashes and stored values equal the feedback P1 (not the TM048 cue P1 `49305bf2…`). Cue keys were consumed (`key_from_cue` true on every clone). That is a real channel, not a no-op.

Those four states are not separable under the frozen criterion `max pairwise L2 > 0.05`:

| Cell | Unique ρ | Max pairwise L2 | Cosine (min pair) | Credit decode |
| --- | --- | --- | --- | --- |
| `scalar_only\|w0` | 1 | 0.0 | 1.0 | 1/4 |
| `scalar_only\|w1` | 1 | 0.0 | 1.0 | 1/4 |
| `action_feedback\|w0` | 4 | 0.01983 | 0.99998 | 1/4 |
| `action_feedback\|w1` | 4 | 0.01457 | 0.99999 | 1/4 |
| `feedback_no_memory\|w0` | 4 | 0.01983 | 0.99998 | 1/4 |
| `feedback_no_memory\|w1` | 4 | 0.01457 | 0.99999 | 1/4 |

Development ceiling still 4/4. Credit ρ still ranks one default action in each world (`h_810668987` on w0, `h_394767965` on w1) — the same defaults as TM048. One generic sensory tick of a unit motor vector is a perturbation of ρ, not an action-identifying state.

## Ladder

`setup_precondition_fail` skipped (decoder 4/4). First behavioral match is `feedback_not_action_separable`. Later rungs were not scored.

This is not a v41 candidate review. Learned K/Q/V is not earned. A stronger or different feedback operator would be a new wall, not a silent change to this freeze.
