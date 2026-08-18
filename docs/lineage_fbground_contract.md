# TM.0.51.FBGROUND contract

**Lab:** TM.0.51.FBGROUND · **Not a v41 candidate.** Product **0.0.004**.

TM050 first-match `states_separate_never_decode` stays frozen. A stronger signal and a new decoder are **not** earned. This wall tests generic developmental grounding of the **existing** `W_act_query` on the **wrapped** post-feedback states TM049 actually writes, using existing plasticity.

## Question

If development teaches only “this internally observed wrapped action state corresponds to this motor action,” on contexts disjoint from tested facts, does held-out wrapped ρ decode 4/4, and does the one-shot episodic chain then complete while no-memory fails and shuffled labels follow the permutation or fail?

## Matched checkpoints

Same world, same birth seed, same `n_dev_repeats`. Only the development protocol differs.

| Protocol | Development |
| --- | --- |
| `reference_only` | current `teach_one` path, feedback flag **off** |
| `feedback_grounded` | same path, flag **on**, so `W_act_query` trains on wrapped `_last_p1` |
| `shuffled_grounding` | same wrapped states (true `motor_vec`), labels permuted by frozen rotation +1 via public `credit_token` |

Grounding cues are `s_dev_*`. Held-out wrap probes use `s_hold_*`. Test facts are the world mapping pairs. Those three sets are disjoint.

Uses the **final wrapped state written to memory** (TM049 `observe` prefix + motor tick + `v_end` + unit P1), not TM050’s isolated insertion ρ.

## After development

1. Freeze slow cortical weights (`eta_act=eta_pred=beta=0`).
2. Assert wrapped decoding on held-out contexts.
3. Run the TM049 one-shot fact protocol (flag on).
4. Reset ρ before canonical recall.
5. `feedback_no_memory`: discard episodes/opaque after the writes.

## Causal chain

ρ_feedback → 4/4 motor decode → S value → 4/4 reinstatement → 4/4 canonical, while no-memory fails. Shuffled grounding follows π or fails; it must not decode true labels 4/4.

If grounding 4/4 but the decoder can no longer express ordinary reference-only development P1s, first-match is `shared_decoder_interference` (not a pass).

## Cells

- **Setup (excluded):** `decoder|w0`, `decoder|w1` — reference_only ordinary development P1s, 4/4
- **Scored:** four arms × two worlds = **8**

## Ladder (setup excluded)

`setup_precondition_fail` → `reference_control_changed` → `heldout_feedback_decode_fail` → `shared_decoder_interference` → `shuffled_not_causal` → `value_projection_fail` → `reinstatement_fail` → `canonical_fail` → `memory_not_necessary` → `feedback_grounding_pass`

## Authorized neural edit

Public `clamp_action(..., credit_token=None)` only. Executed `motor_vec` comes from `token`. Credit / episode / `W_act_query` label comes from `credit_token` or `token`. No new matrix. No feedback hyperparameter. Not a GenomeConfig field. Not an `ACT_RECALL_MODE`.

## Refuse

K/Q/V, new decoder, fitted tick count, scaling `motor_vec`, TM049/TM050 rerun, v41, product-earn, runner-manufactured ρ/k/v, copying handles into S.
