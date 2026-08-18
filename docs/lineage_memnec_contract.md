# TM.0.45.MEMNEC contract

**Lab:** TM.0.45.MEMNEC · **Not a v41 candidate.** Product **0.0.004**.

TM044 first-match `memory_not_necessary` stays frozen. This wall does **not** tune \(W_k,W_q,W_v\). It asks where persistent information is actually required.

## Protocol

Reset ρ (and leftover P1) before recall. Then, on the current implementation:

- arbitrary cue→handle bindings at 2, 4, and 8 cues (2 handles)
- conditions: immediate, delayed (4 idle ticks), distractor, revision
- arms: `symbolic_oracle`, observational `learned_projection`, `no_persistent_memory`
- record `memory_path`, `motor_path`, `scoring_address_source`, and scores before and after reinstatement

## Target

The first frozen cell where the oracle succeeds and `no_persistent_memory` fails. That cell is the earned site for any later projection-learning repair. Learned-arm accuracy is observational and is not a gate.

## Refuse

K/Q/V edits, TM044 rerun, v41 lock, `joint_socp.py` edits, product-earn.
