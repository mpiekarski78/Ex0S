# TM.0.55.DRIFT contract

**Lab:** TM.0.55.DRIFT · **Not a v41 candidate.** Product **0.0.004**. Diagnostic only. No neural edit. Do not install W\*. Do not retune `EPISODE_MATCH_L2` (0.05). Do not force cue context into the value code. Do not treat frozen wraps as canonical.

TM054 first-match `state_generator_mismatch` stays frozen. The addendum reading stays frozen: canonical generator is write-time last P1. Frozen wraps remain diagnostic predictions of how the current cortex would encode a similar future event.

## Law

\[
k_t = W_k\,\rho_{\mathrm{cue},t},\qquad v_t = \mathrm{unit}(\rho_{\mathrm{post\text{-}feedback},t})
\]

captured and stored at event time. Never regenerate historical values from a later checkpoint.

## Question

Does the organism naturally maintain an action-invariant value code online, or does write-time \(v\) drift so that global consolidation can only repair history after every event?

The frozen post-checkpoint family showed that an action-invariant value code is possible. It cannot replace historical write-time values.

## Method

Follow one continuous, action-balanced developmental trajectory. Capture actual write-time values — never regenerate them. Train diagnostic \(W_N^*\) using only the first \(N\) historical writes plus ordinary reference-action constraints. Test it on later, genuinely future write-time values. Train a suffix \(W^*\) on late writes and test it on early writes. Discard every \(W^*\).

`n_dev_repeats` stays 4 for the reference-only arm. `n_online_repeats` is this wall’s trajectory length, not an increase of the frozen TM046 constant. \(N\) is a diagnostic prefix index, not an organism constant.

## Ladder (setup excluded)

`setup_precondition_fail` → `prefix_infeasible` → `incompatible_action_clusters` → `catastrophic_representational_migration` → `write_must_be_included` → `developmental_coordinate_drift` → `grounding_consolidation_plausible`

| Result | Interpretation |
| --- | --- |
| Past writes generalize to future writes | Grounding consolidation is plausible |
| Every new write must be included before it decodes | Representation is drifting; global consolidation is only repairing history after every event |
| Same action forms multiple incompatible clusters | Value formation needs context isolation or a learned invariant \(W_v\) |
| Old writes decode but new writes do not | Developmental coordinate drift |
| New writes decode while old writes fail | Catastrophic representational migration; explicit memory reconsolidation would be required |

## Refuse

Install W\*, regenerate wraps, treat frozen wrap as canonical, retune 0.05, force context into \(v\), increase frozen `n_dev_repeats`, extend joint SOCP, second decoder, K/Q/V, rewrite TM053/TM054, v41, product-earn.
