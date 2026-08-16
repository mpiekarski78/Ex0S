# Sequence-evidence contract — TM.0.18.SEQUENCE Phase B

**Lab:** TM.0.18.SEQUENCE.MECH

**Product under test:** Ex0S **0.0.004**

**Flags:** `earned_next=false`, `ex0s=null`

**Prereg:** [`sequence_mech.prereg.lock`](sequence_mech.prereg.lock)

After Phase A shows frozen SYMBOLWORLD can select a word but cannot construct utterances, this contract defines one general sequence candidate.

## Claim

> An opt-in recipe may author raw sequence-step rows into `experience_sequence` from exact `observe_sequence_step` tuples over factorized context atoms, input symbols, and output prefix, and at use time recompute unique `next_operation` ∈ {emit, stop} (with `next_symbol` on emit) so `emit_sequence` constructs a variable-length utterance or returns atomic HOLD. No scene IDs, grammar slots, menus, or complete-response lookup. STOP placement is evidenced. Cap=64 does not reveal expected length.

## Context rule

`context_atoms` is a factorized collection of observable or already-justified grounded atoms. Forbidden: `scene_184`, `hash(complete_scene)`, role labels, expected answer, serialized response, scorer state.

## Channel

| Field | Meaning |
|---|---|
| `context_atoms` | sorted list of opaque atoms |
| `input_symbols` | ordered input/question tokens |
| `prefix` | already-emitted output tokens |
| `next_operation` | `emit` or `stop` |
| `next_symbol` | token if emit; empty if stop |
| `result` | `success` / `failure` / `correction` |

## Emission

Atomic: any internal failure → external HOLD (no partial sequence). Ambiguous emit/stop or next token → HOLD. Cap without unique STOP → HOLD. Safety cap = 64 while scored lengths ≤ 16.

## Isolation

Sequence may use derived grounded interpretation but must not query fingerprint/continuity as sentence source. Required forks as in mech prereg.

## Unit cells

U0 flag-off; U1 reject; U2 one-word earn; U3 tie HOLD; U4 strip; U5 donor.

## Pin order

Unscored smoke → `sequence.candidate.lock` → scored cells → E-life → capacity → dialogue (dialogue prereg already frozen).

## Composition

When exact `(context_atoms, input, prefix)` evidence is absent, emit may transfer order from a uniquely nearest taught complete sequence whose context differs by **exactly one** atom substitution, mapping tokens through grounding. Multi-atom remaps and equal-distance conflicting templates → HOLD.
