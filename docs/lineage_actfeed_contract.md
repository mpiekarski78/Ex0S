# TM.0.49.ACTFEED contract

**Lab:** TM.0.49.ACTFEED · **Not a v41 candidate.** Product **0.0.004**.

TM048 first-match `credit_action_information_absent` stays frozen. This wall freezes one exact action-feedback transition and its lifecycle **before** the neural edit.

## Cells

- **Setup (excluded from behavioral first-match):** `decoder|w0`, `decoder|w1`
- **Scored:** three arms × two worlds = **6** cells

| Arm | Meaning |
| --- | --- |
| `scalar_only` | flag off; must reproduce TM048 collapse |
| `action_feedback` | flag on; generic `_sensory_tick` channel |
| `feedback_no_memory` | flag on; persistent write discarded after credit |

## Exact transition (frozen)

Reuse the named existing cortex transition **`_sensory_tick`**, the same operation that injects cue symbols while building development-reference states.

\[
x = \texttt{\_x\_tick}(\texttt{motor\_vec},\,\mathrm{body},\,\texttt{same\_ix})
\]
\[
\tilde\rho = \tanh\bigl((W_{\mathrm{rec}}\odot M)\rho + W_{\mathrm{in}}x + b + W_{\mathrm{body}}\,\mathrm{body}\bigr)
\]
\[
\rho \leftarrow \tilde\rho \quad (\texttt{record\_sensory=True}:\ \mathrm{no}\ \texttt{motor\_persist\_p}\ \mathrm{mix})
\]

Pinned:

| Item | Value |
| --- | --- |
| Named op | `_sensory_tick` |
| Recurrent ticks | **1** |
| Activation | `tanh` |
| `record_sensory` | `True` (no persist mix) |
| `motor_vec` entry | as `_sensory_tick`'s `injected` argument |
| Projection | existing `_x_tick` → `W_in` only |
| `motor_vec` space | unit `d_sym`, same as `_vocab_vec` (already from `bind_actuators`) |
| Value P1 | existing `_unit_or_zero(ρ)` after the tick sequence, same as `_last_p1` |
| New matrix | **forbidden** |
| Feedback hyperparameter | **forbidden** |

Insertion on the **credit/outcome** `observe()`, when the flag is on and a pending `motor_vec` is legal:

1. `_sensory_tick(start)`
2. one `_sensory_tick(_vocab_vec(u))` per cue symbol
3. **one** `_sensory_tick(pending.motor_vec, body, same_ix, record_sensory=True)`
4. `_sensory_tick(v_end)` then `_last_p1 = unit(ρ)` ← \(\rho_{\mathrm{feedback}}\)
5. `_sensory_tick(s_t)`

Storage: \(k=W_k\rho_{\mathrm{cue}}\) from the **consumed prior-cue** `pending.key_rho`, not from this observe's post-feedback key. \(v=W_v\rho_{\mathrm{feedback}}\) from that `_last_p1`.

Flag off: no extra tick (TM048).

## Lifecycle (frozen)

- Cue state (`pending.key_rho` / `pending.rho_p1`) is captured on the cue `observe()` and **consumed once** at credit.
- Pending `motor_vec` is **consumed once** and cleared after the action tick.
- Credit without a pending action (`pending is None` or `motor_vec` missing/nonfinite/wrong dim) **fails closed**: no action tick, no episode write.
- Duplicate credit cannot reuse stale feedback (cleared `motor_vec` and consumed cue key).
- Teacher clamp uses public `clamp_action` only. The runner never writes `_pending`, `_last_p1`, `_last_key_rho`, episodes, or S.
- Checkpoint boundaries are whole public steps only: after cue `observe()`, after `clamp_action`, after credit `observe()`. The action tick and write are atomic inside one `observe()`. The flag and pending (including `motor_vec`, `key_rho`) survive checkpoints.
- Zero/negative advantage may still produce feedback ρ; it **cannot** write the rewarded episode (`adv > ELIG_EPS` required to write).

## Ladder (behavioral first-match; setup excluded)

`setup_precondition_fail` → `feedback_not_action_separable` → `feedback_rho_fail` → `value_projection_fail` → `reinstatement_fail` → `canonical_fail` → `scalar_control_changed` → `memory_not_necessary` → `action_feedback_pass`

Learned opaque addressing is **not** earned on this wall.

## Refuse

K/Q/V tuning, new decoder, new fitted matrix, feedback-specific hyperparameter, `ACT_RECALL_MODE` / `GenomeConfig` flag, TM048 rerun, v41, product-earn, runner-manufactured ρ/k/v, copying the handle into S.
