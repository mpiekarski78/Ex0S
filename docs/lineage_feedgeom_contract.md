# TM.0.50.FEEDGEOM contract

**Lab:** TM.0.50.FEEDGEOM · **Not a v41 candidate.** Product **0.0.004**. Diagnostic only.

TM049 first-match `feedback_not_action_separable` stays frozen. This wall does **not** edit the cortex, fit a tick count, scale `motor_vec`, retrain the decoder, or retune `EPISODE_MATCH_L2`.

## TM049 interpretation (frozen)

Action information is **present**: four unique feedback hashes, and stored P1 equals feedback P1. It is **weak geometrically** (max L2 0.0146–0.0198, below the 0.05 diagnostic floor). It is **behaviorally unreadable**: motor decoding remains 1/4. The 0.05 floor is a frozen diagnostic boundary, not proof that states below it contain no information.

## Question

Using the same development checkpoint and the existing `_sensory_tick(motor_vec)` transition, at which stage, tick in `{0,1,2,4,8,16}`, and context (cue ρ vs reset ρ) does action-dependent geometry become separable and motor-decodable?

## Cells

- **Setup (excluded from behavioral first-match):** `decoder|w0`, `decoder|w1`
- **Scored:** two contexts × two worlds = **4** cells

| Cell prefix | Meaning |
| --- | --- |
| `cue` | action ticks start from TM049 insertion ρ (cue observe + start/symbol prefix, no motor tick yet) |
| `neutral` | the same ticks start from `reset_rho()` |

Shared outcome body across actions (TM049 already produced one body). `same_ix=0`. Flag stays off; the runner applies the named existing tick on clones.

## Measurement (frozen)

For each action and tick \(t\in\{0,1,2,4,8,16\}\), on a side-effect-free clone:

| Stage | What |
| --- | --- |
| `motor_vec` | unit `d_sym` actuator vector |
| `x_tick` | existing `_x_tick(motor_vec, body, same_ix)` |
| `w_in_x` | existing `W_in x` |
| `preactivation` | `(W_rec ⊙ M)ρ + W_in x + b + W_body body` |
| `rho_t` | post-tanh ρ after *t* motor ticks (`t=0` is the start state) |
| `p1` | existing `_unit_or_zero(ρ_t)` |
| `motor_decode` | existing motor decoder winner and margin on P1 |
| `tanh_saturation` | mean \|pre\|, fraction of units with \|tanh(pre)\| > 0.99 |
| `dev_reference` | distance from P1 to that action's development-reference P1 |

Separability diagnostic: `n_unique == 4` **and** max pairwise L2 > frozen `EPISODE_MATCH_L2` (0.05). Unique hashes alone are not separability. The floor is not retuned.

The tick grid is a measurement schedule, not a fitted organism hyperparameter. Finding that later ticks decode does **not** authorize changing TM049's `n_ticks=1`.

## Ladder (behavioral first-match; setup excluded)

`setup_precondition_fail` → `collapse_before_recurrence` → `input_separates_tanh_compresses` → `later_ticks_decode` → `neutral_passes_cue_fails` → `states_separate_never_decode` → `geometry_unresolved`

| Code | Interpretation |
| --- | --- |
| `collapse_before_recurrence` | Action transduction is inadequate (`motor_vec` separates; `x_tick` or `W_in x` does not, at cue `t=1`) |
| `input_separates_tanh_compresses` | Input separates, tanh compresses (`W_in x` separates at cue `t=1`; `ρ_t` does not) |
| `later_ticks_decode` | Temporal integration is missing (cue decode is not 4/4 at `t=1`, is 4/4 at some later grid tick) |
| `neutral_passes_cue_fails` | Cue-state interference (neutral decode 4/4 at some t; cue never 4/4) |
| `states_separate_never_decode` | Feedback and motor manifolds are misaligned (cue `ρ`/`p1` separable at some t; decode never 4/4) |
| `geometry_unresolved` | None of the patterns match |

All pattern flags are recorded even when first-match is an earlier row. No mechanism is earned. No v41.

## Refuse

Neural edit, K/Q/V, new decoder, fitted tick count, scaling `motor_vec`, retuning 0.05, TM049 rerun, v41, product-earn, runner-manufactured storage, copying the handle into S.
