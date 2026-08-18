# TM.0.57.DUAL results

## Decision: `storage_integrity_failure`

Product **0.0.004**. Canonical law unchanged. Floor **0.05** was not retuned. No v41 lock. Every \(W^*\) was discarded.

TM056 first-match remains `replaced_under_write_law`. The addendum interpretation remains `tm052_selected_surviving_records`. REST and live-state telemetry stay innocent.

DEV ran once on clean `2eca88e590489dd1f0cfb2cf3326fd882f868eca`. Frozen runner SHA `1f1ee4b8d4d2da7893622d8692a91b3912ed7130f9a868dffe02fc19d5cd8f61`.

\(v_t\) is the value offered to `_episode_write`. Resident value is what S retains after append, eviction, or refresh.

## What was recovered

Ordinary reference-action decode was 4/4 on both worlds. Setup is not the first-match.

World 0 write 17 is a P1-neighbour **replace** of action `h_619319047` / cue `s_at_0_3_2` by a different action `h_431912685` / cue `s_at_0_4_0`. The new \(v_t\) was stored. That is a storage-integrity failure: content-addressed replacement crossed action identity.

| World | append | evict | same-cue refresh | different-cue same-action refresh | different-action replace |
| --- | --- | --- | --- | --- | --- |
| 0 | 8 | 20 | 0 | 3 | 1 |
| 1 | 8 | 21 | 0 | 3 | 0 |

World 1 did not cross action identity. Both worlds still collapsed cue addressing: in-place refresh kept the old resident P1 and attached a later unique cue of the same action. Eviction is the majority outcome and is capacity lifecycle, not the first-match.

The scored curve still ran and is on file. All 16 attempt and resident cells are `n_prefix_only` (`future_ok=false`). That is not the decision. First-match already shows the inherited P1 replacement law is not compatible with opaque key/value memory on this trajectory. `p1_replacement_law_compatible_with_opaque_kv` is false.

## What this is not

Not a license to retune 0.05, not a rewrite of TM056, not an installed oracle, not a v41 candidate, not a claim that attempts or residents generalized.
