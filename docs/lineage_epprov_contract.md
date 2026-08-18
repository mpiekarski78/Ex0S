# TM.0.56.EPPROV contract

**Lab:** TM.0.56.EPPROV · **Not a v41 candidate.** Product **0.0.004**. Diagnostic only. No neural edit. Do not install W\*. Do not retune `EPISODE_MATCH_L2` (0.05). Do not reconsider the write-time law.

TM055 first-match `setup_precondition_fail` stays frozen. It is a valid setup stop. Canonical \(v_t\) remains the value passed into the actual event-time episode write.

There cannot be two passive stored snapshots. An episode is either written once and unchanged, explicitly replaced, explicitly reconsolidated, or accidentally mutated / incorrectly selected by telemetry. “Last episode after REST” is not automatically the historical record written during grounding.

## Question

On the pinned TM052 reconstruction, what happened to each grounding write between `_episode_write` and the TM052 handle-scan measurement?

Identify the target through write provenance — not `episodes[-1]`, handle lookup, or live `_last_p1`. Those selectors may be recorded only as comparators.

## Record (every grounding event)

just-written record identity/index;
key, value/P1 and full-record hashes at `_episode_write`;
episode count before/after;
hashes after `ground_one`;
after the later `s_t` tick;
after each REST stage;
after evaluation preparation;
live `_last_p1` separately;
any replacement or in-place mutation and its explicit cause.

## Ladder (setup excluded)

`setup_precondition_fail` → `reconstruction_setup_mismatch` → `pinned_live_last_p1` → `p1_mutated_during_rest` → `replaced_under_write_law` → `later_record_selected` → `tm052_measured_wrong_record`

| Observation | Interpretation |
| --- | --- |
| Same record and P1 throughout | TM052 selected/measured the wrong record |
| Original preserved but a later record selected | Record-selection bug |
| Original replaced under the 0.05 write law | Lifecycle/replacement, not drift |
| P1 mutated during REST | Reconsolidation if authorized; provenance violation otherwise |
| TM052 pinned live `_last_p1` | Measurement-boundary error |
| Write itself differs | Reconstruction/setup mismatch |

## Refuse

Install W\*, retune 0.05, reconsider the law, treat REST-last as automatically historical, identify the target by `episodes[-1]` / handle lookup / live P1, rewrite TM055, v41, product-earn.
