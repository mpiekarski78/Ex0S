# TM.0.57.DUAL contract

**Lab:** TM.0.57.DUAL · **Not a v41 candidate.** Product **0.0.004**. Diagnostic only. No neural edit. Do not install W\*. Do not retune `EPISODE_MATCH_L2` (0.05). Do not reconsider the write-time law.

TM056 first-match `replaced_under_write_law` stays frozen. REST and live-state telemetry are innocent. TM052 selected surviving records rather than final write identities.

## Distinction

- \(v_t\): the event-time value offered to `_episode_write`. Historical experience even when not retained.
- **Resident value:** what S actually retains after append, eviction, or refresh. When \(d \le 0.05\), \(v_t\) is not persistent memory.

## Question

Do attempted write-time values generalize forward, do organism-available residents generalize forward, and does the inherited P1 replacement law preserve opaque key/value identity?

## Method

One continuous action-balanced trajectory. Unique cues per event. Capture every write receipt at `_episode_write`. Do not use `episodes[-1]`, handle lookup, or live `_last_p1` as the target.

Chronological datasets:

1. **All write attempts** — diagnostic ceiling for representational stability.
2. **Organism-available residents** — the store after each prefix; what consolidation could actually learn from S.

Train diagnostic \(W_N^*\) on the first \(N\) items of each tape plus ordinary reference-action constraints. Test later attempts, and the final resident store. Discard every \(W^*\).

## Receipts

attempted \(v_t\) hash;
outcome: append, evict, refresh, reject;
affected record identity;
resident value before/after;
cue key and action on both sides;
whether refresh crosses cue or action identity.

| Refresh | Meaning |
| --- | --- |
| Same cue/action | Ordinary deduplication |
| Different cue, same action | Acceptable for grounding examples; cannot represent distinct cue-addressed facts |
| Different action | Storage-integrity failure |
| Eviction | Capacity lifecycle, not representation drift |

## Ladder (setup excluded)

`setup_precondition_fail` → `storage_integrity_failure` → `cue_addressing_collapsed` → `prefix_infeasible` → `representation_drift` → `storage_selection_capacity_wall` → `grounding_consolidation_plausible`

| Result | Interpretation |
| --- | --- |
| Attempts generalize, residents do not | Storage-selection/capacity wall |
| Attempts themselves do not generalize | Representation drift |
| Both generalize | Grounding consolidation is plausible |

## Refuse

Change the 0.05 law, install W\*, treat handle-scan as \(v_t\), rewrite TM056, v41, product-earn.
