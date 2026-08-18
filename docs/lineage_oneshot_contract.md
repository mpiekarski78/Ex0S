# TM.0.46.ONESHOT contract

**Lab:** TM.0.46.ONESHOT · **Not a v41 candidate.** Product **0.0.004**.

TM045 showed why an ordinary association sweep cannot earn neural memory: at 2/4 cues the facts are already in slow `W_act_query`, and at 8 cues both arms hit the same actuator-capacity wall. This wall isolates **timescales**, not cue count.

## Protocol

1. Build a **common developed checkpoint**: generic sensory processing plus a motor decoder that can express every bound action from a valid reinstated state. Development cues are disjoint from test facts. Then clear episodic/opaque S.
2. **Freeze** slow cortical `W_act_query` for the diagnostic fact phase (`freeze_plasticity`).
3. Teach **many arbitrary facts once**, including a rapid revision. Permit only episodic/opaque S writes.
4. Reset ρ, then probe with delay and distractors through `event_memory_scores`.

## Arms

- `symbolic_oracle` — episodic completion ceiling
- `opaque_projection` — current opaque S with birth K/Q/V, observational, not tuned
- `no_persistent_memory` — neither S nor slow cortex may hold the test facts
- `slow_cortex_enabled` — observational consolidation control

## Target

The earned cell is oracle success and `no_persistent_memory` failure. Only then is opaque K/Q/V a legitimate target.

If the decoder precondition fails, stop. If the decoder is fine but the oracle still cannot pass one-shot facts, the earned site is the **generic reinstatement-to-motor interface**, not learned addressing.

## Refuse

K/Q/V edits, another cue-count grid, TM044/TM045 reruns, v41 lock, `joint_socp.py` edits, product-earn.
