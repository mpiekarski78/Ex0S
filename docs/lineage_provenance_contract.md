# TM.0.54.PROV contract

**Lab:** TM.0.54.PROV · **Not a v41 candidate.** Product **0.0.004**. Diagnostic only. No neural edit. Do not install W\*. Do not retune `EPISODE_MATCH_L2` (0.05). Do not force cue context into the value code.

TM053 first-match `coverage_generalizes` stays frozen. The addendum reading stays frozen: the measurement is invalidated because support states were duplicates, not because an action-invariant value is a defect. `k` is cue/context identity; `v` is action experience. A stable per-action value would be ideal.

## Question

Which state-generation procedure matches the organism’s real credit-and-write lifecycle?

TM053 cannot answer coverage because it did not reproduce TM052’s states. Before another coverage wall, compare TM052 and TM053-style generation at every state boundary on the **pinned TM052 reconstruction**, then gate N=1 train and held-out hashes against those pins **before scoring anything**.

## Boundaries

parent checkpoint, W, ρ, vocabulary and registry hashes;
cue-state hash before feedback;
interaction index and body state;
`motor_vec`, `_x_tick`, and `W_in @ x`;
ρ immediately after the feedback tick;
state after `v_end`;
state after `s_t`;
final wrapped P1;
reset and consume-once lifecycle.

## Gate

Canonical N=1 training hashes and held-out hashes must equal pinned TM052 hashes. If they do not: `state_generator_mismatch`. No SOCP, no coverage scoring.

## Fork (only if hashes match)

| Result | Then |
| --- | --- |
| Four action-invariant values | Excellent; one exemplar per action is enough to test generic grounding consolidation |
| Context-varying values | Rerun a genuine coverage/generalization curve on these states |
| Pipelines differ from an unintended runner/reset sequence | Fix only the future canonical probe; leave historical walls intact |

## Refuse

Rewrite TM053, retune 0.05, force context into v, install W\*, extend v40, second decoder, K/Q/V, v41, product-earn.
