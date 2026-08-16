# Grounding-evidence contract — SYMBOLWORLD Phase B

**Lab:** TM.0.17.SYMBOLWORLD.GROUND

**Product under test:** Ex0S **0.0.004**

**Flags:** `earned_next=false`, `ex0s=null`

**Prereg:** [`symbol_ground.prereg.lock`](symbol_ground.prereg.lock)

After Phase A freezes an unfamiliar symbolic world and shows PERSIST-on cannot ground words, this contract defines one general grounding candidate. Alias fingerprints and continuity marks remain separate tracks and never become word meaning.

## Claim

> An opt-in recipe may author raw co-occurrence rows into `experience_grounding` from exact `observe_symbol_ground` tuples and, at use time only, recompute evidence-weighted bindings between utterance symbols and paired world tokens so a unique winner among offered choices can be selected. The same machinery serves nouns, verbs, properties and roles — no POS-specific learners. Ties and insufficient support produce HOLD. Meaning is never stored as a synonym map.

## Frozen evidence rule

- Rows are co-occurrence evidence only: `{symbol, paired, trial_id, result}`.
- `result`: `success` / `correction` add +1 support for the paired token; `failure` subtracts 1.
- For each symbol, permission requires a unique paired token with net support ≥ `min_support` (2) and strictly above any runner-up.
- Selection is among **world-offered choices** (actions, entities, features, compounds, or bounded word menus). Free sentence generation is out of scope.
- Scorer knows latent meanings; Ex0S never receives `ball = feature_17` equations.
- Fingerprint and continuity rows are inaccessible as word meaning.

## Channel contract

| Allowed field | Meaning |
|---|---|
| `symbol` | Opaque utterance token |
| `paired` | Co-visible world token (feature / entity / action / compound) |
| `trial_id` | Opaque trial grouping |
| `result` | `success` / `failure` / `correction` |

Forbidden: POS tags, gloss, english, meaning, same_as, planted role labels that smuggle grammar class.

## Unit cells

| Cell | Evidence | Honest outcome |
|---|---|---|
| U0 flag off | Valid rows with flag false | write nothing; probe HOLD |
| U1 reject | Malformed ABI | reject; write nothing |
| U2 earn | Unique support ≥ 2 | select paired |
| U3 equalize | Two tied pairings | HOLD |
| U4 strip | Earn then strip rows | HOLD |
| U5 donor | Swap grounding rows in S | behavior follows donor |

## Later

Score the continuous developmental life S0–S10 on the frozen candidate (Phase C). No mechanism changes between stages. Product stamp stays 0.0.004.
