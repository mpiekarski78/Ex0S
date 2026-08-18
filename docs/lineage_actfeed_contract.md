# TM.0.49.ACTFEED contract

**Lab:** TM.0.49.ACTFEED · **Not a v41 candidate.** Product **0.0.004**.

TM048 first-match `credit_action_information_absent` stays frozen. Post-credit ρ is identical across credited actions, so no downstream projection can recover which action was credited. This wall freezes a **generic action-feedback / efference-copy channel** before the neural edit.

## Critical law

Use two temporally distinct cortical states:

\[
k = W_k\,\rho_{\mathrm{cue}},\qquad
\rho_{\mathrm{feedback}} = F(\rho_{\mathrm{cue}}, x_{\mathrm{action}}, r),\qquad
v = W_v\,\rho_{\mathrm{feedback}}
\]

The key comes from the **pre-feedback cue state**. The value comes from the **post-feedback action state**. Otherwise the stored key may describe the action-feedback event and fail to match the cue at recall.

\(x_{\mathrm{action}}\) enters through **one** generic action-observation channel:

- self-generated action → efference copy (pending `motor_vec`)
- teacher-provided correct action → demonstrated-action feedback (same pending `motor_vec` after clamp)
- scalar reward gates credit/write; it does not identify the action

The runner may present the demonstrated action as environmental experience (existing `clamp_action`). It must **not** construct \(\rho\), \(k\), or \(v\), and must **not** copy the handle into S.

## Flag

`action_feedback_enabled` defaults **off**, survives checkpoints, and is **not** an `ACT_RECALL_MODE` and **not** a `GenomeConfig` field. Slow `W_act_query` stays frozen. Symbolic addressing is the diagnostic oracle. SOCP off. \(W_k,W_q,W_v\) unchanged.

## Matched arms

| Arm | Meaning |
| --- | --- |
| `scalar_only` | frozen TM048 behavior (flag off) |
| `action_feedback` | new generic feedback channel |
| `feedback_no_memory` | feedback occurs; persistent write is discarded |

## Required, in order

1. Different actions produce distinguishable feedback ρ.
2. Feedback ρ ranks all four actions correctly.
3. Stored opaque/episode values preserve that ranking.
4. Exact reinstatement preserves it.
5. Canonical episodic behavior passes.
6. `scalar_only` reproduces the default collapse.
7. `feedback_no_memory` fails after reset.

If this passes, the organism has a complete episodic loop: cue → experienced action/outcome → cortical trace → S → reinstatement → action. Learned opaque addressing is **not** earned on this wall.

## Refuse

K/Q/V tuning, new decoder, TM048 rerun, v41 lock, product-earn, runner-manufactured ρ/k/v, copying the handle into S.
