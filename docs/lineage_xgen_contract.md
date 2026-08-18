# TM.0.62.XGEN contract

**Lab:** TM.0.62.XGEN · **Not a v41 candidate.** Product **0.0.004**. Diagnostic only. No neural edit. No SOCP installation. Do not install W\*. Do not redesign K/Q/V. Do not retune `EPISODE_MATCH_L2` (0.05). Do not rerun or edit TM060 or TM061.

TM061 first-match `transport_restores_decoding` stays frozen. Architectural conclusion remains none. A 64×64 linear map on 16 paired states is underdetermined; 16/16 can be interpolation, not a generic transport law. Orthogonal failure only showed that the in-sample map includes scaling/shearing.

## Holdout

Fit transport on a chronological subset that contains all four organism-emitted action roles. Score later aligned contexts that were not used to fit. Correspondence is (emitted action, occurrence index). Runner handle strings are diagnostic names only, never record identity.

Identity, orthogonal, diagonal, affine, low-rank, ridge, and minimum-norm linear are separate discarded ceilings. Unrestricted in-sample linear is the interpolation ceiling, not a generalization claim. Every map is discarded. Parent `W_act_query` hashes stay unchanged.

## Ladder (setup excluded)

`setup_precondition_fail` → `observer_used_runner_provenance` → `full_oracle_infeasible` → `prefix_baseline_fail` → `later_values_unreadable` → `tm061_in_sample_not_reproduced` → `only_unrestricted_interpolates` → `action_conditioned_transport` → `constrained_transport_generalizes` → `shared_map_transfers`

| Result | Interpretation |
| --- | --- |
| Simple constrained transport generalizes | Stable-coordinate or online alignment becomes plausible |
| Only unrestricted linear fits its training pairs | TM061 was interpolation; return to representation-learning hypotheses |
| Transport generalizes only within an action | Action-conditioned manifolds, not one generic coordinate drift |
| One map transfers across actions and held-out contexts | A small alignment organ may be enough; no major Miconi-style jump yet |

## Refuse

Neural edit, SOCP install, install W\*, K/Q/V redesign, treat TM061 as a generic transport law, predict organism ids, retune 0.05, rewrite TM060/TM061, v41, product-earn.
