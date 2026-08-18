# TM.0.52.SHAREFEAS contract

**Lab:** TM.0.52.SHAREFEAS · **Not a v41 candidate.** Product **0.0.004**. Diagnostic only. No neural edit. Do not install W\*.

TM051 first-match `heldout_feedback_decode_fail` stays frozen. A new decoder is **not** automatically earned. This wall asks whether one shared `W_act_query` can satisfy both the ordinary development-reference manifold and the wrapped action-feedback manifold.

## Locked TM051 telemetry (audit)

The TM051 DEV lock records:

- setup `decoder|w{0,1}`: reference-only development P1s decode **4/4**
- `feedback_grounded` held-out wrap: **1/4**, four unique P1 hashes
- ordinary reference P1s after grounding: **2/4** and **1/4**

It does **not** record whether grounded development decoded its own wrapped training states. This wall reconstructs those states with the frozen TM051 runner and records that decode before any SOCP.

## Question

Using the existing joint min-change SOCP geometry and the organism predicate, without installing weights, is there a shared W\* on:

1. wrapped grounding-training states only
2. reference development P1s + wrapped training states
3. reference + training + held-out wrapped feedback states

## Ceilings (side-effect-free)

W0 is the grounded checkpoint `W_act_query`. Constraints are the existing rival-difference rows (`d = v_h − v_r`, `x` unit P1, `τ = ACT_MARGIN_FLOOR`). Solver output W\* is hashed and discarded. The parent hash must not move.

| Ceiling | States |
| --- | --- |
| `wrapped_train` | four wrapped training P1s written during TM051 grounding |
| `train_ref` | those plus four reference-only development P1s |
| `full_oracle` | those plus four held-out wrapped feedback P1s |

## Ladder (setup excluded)

`setup_precondition_fail` → `training_infeasible` → `reference_feedback_conflict` → `context_entangled` → `shared_W_star_satisfies`

| Result | Conclusion |
| --- | --- |
| Even training set jointly infeasible | Feedback representation or shared decoder capacity is inadequate |
| Training feasible, held-out fails even with trained-only W\* / full joint infeasible | Feedback representation is context-entangled |
| Reference and feedback sets conflict (`train_ref` infeasible) | A separate feedback-to-action alignment path may be earned |
| Full oracle set feasible | Do not add a decoder; investigate generic consolidation for grounding |

TM051 training-wrap decode (reconstructed, no SOCP) is telemetry: never-fit implies optimization; train-fit with held-out 1/4 implies generalization.

## Refuse

Install W\*, extend v40 SOCP, increase `n_dev_repeats`, second decoder, K/Q/V, TM051 rerun rewrite, v41, product-earn.
