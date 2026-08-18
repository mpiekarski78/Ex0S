# TM.0.48.CREDITINFO contract

**Lab:** TM.0.48.CREDITINFO · **Not a v41 candidate.** Product **0.0.004**.

TM047 first-match `credit_rho_fail` stays frozen. One-shot credit ρ is off the decoder manifold while retrieval and injection stay intact. This wall asks whether the **credited action ever enters ρ**. It does **not** tune \(W_k,W_q,W_v\), does **not** train a new decoder, and does **not** add an action-feedback channel.

## Protocol

Reconstruct the TM046 common checkpoint. Freeze slow `W_act_query`. From that identical pre-credit snapshot, clone four times and credit the **same cue** with four different actions through the existing `teach_one` path (observe cue → clamp action → observe outcome). Record:

- ρ immediately before credit
- ρ immediately after credit
- pairwise L2/cosine and hashes across credited actions
- motor ranking from each credit ρ
- projected stored value \(W_v\rho\) and the episode P1 actually written to S
- whether varying only the credited action changes ρ

Use development action states as the positive ceiling. The runner must not copy an action vector into S, must not select a memory row, and must not manufacture the stored value.

## Target

| Code | When |
| --- | --- |
| `credit_action_information_absent` | different credited actions produce identical or action-indistinguishable ρ |
| `credit_trace_present_not_decodable` | states differ, but the generic motor decoder cannot distinguish them |
| `value_projection_loss` | credit ρ is 4/4, but \(W_v\rho\) is not |
| `credit_information_pass` | credit ρ and projected value both decode all four actions |

If action information is absent, the earned *next* mechanism is a generic action-feedback / efference-copy channel. This wall does not implement it.

## Refuse

K/Q/V edits, new decoder, action-feedback implementation, TM046/TM047 reruns, v41 lock, product-earn, copying the action into S.
