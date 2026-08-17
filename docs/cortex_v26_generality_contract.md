# TM.0.23.CORTEX.GENERALITY.v26

**Lab:** TM.0.23.CORTEX.GENERALITY.v26  
**Product:** Ex0S **0.0.004** · `earned_next=false` · `ex0s=null`  
**Under test:** frozen [`cortex.candidate.v26.lock`](cortex.candidate.lock) — candidate **immutable**  
**Refuse:** neural edits, scorer edits, FULLDEV.R7 reveal, product stamp

This audit is scored once on live v26. Freeze this contract and the runner **before** scoring. Instrumentation must not consume organism RNG or perturb trajectories.

## Claim under test

Continuation, refractory, STOP, and familiarity on v26 are **generic learned dynamics**, not scripted phrase machinery. Isolated D5–D7 greens do not transfer to FULLDEV.R7 unless every control below is green.

## Controls (all required)

| ID | Pass rule |
|----|-----------|
| `G1_no_scripted_phrase_machinery` | Source of `three_memory/neural_cortex.py` and `three_memory/cortex_memory.py` has **zero** hits on the frozen forbidden identifiers / regexes below |
| `G2_cross_modal_transfer` | After a familiar non-speech cue, HOLD-after-ACT rate is within 0.25 of HOLD-after-EMIT and both are ≥ 0.30. Need ≥ 4 EMITs and ≥ 4 ACTs in the sample |
| `G3_non_echo_response` | After consequence teaching *heard A → emit B*, probes of A yield `emit_B > echo_A` and `emit_B ≥ 3` on 20 probes. Echo of A is not the answer |
| `G4_order_length_counterfactuals` | All four subtests green: renamed twin follows own tokens; reverse probe does not copy the reverse; distractor is not replayed; unseen length is not copied from the probe |
| `G5_stop_evidence` | Two lives, same 4-token sensory phrase, different STOP consequences (boundary at 2 vs 4). Mean emitted length at STOP differs by ≥ 1.0 and each life is closer to its taught boundary than to the other |
| `G6_neophobia_provenance` | Familiarity from neural/S experience: taught token nonhold exceeds unknown by ≥ 0.15; `reset_cortex` restores unknown-like HOLD (≥12/20); donor checkpoint restores familiarity; no `known_chunks` identifier |
| `G7_ablations` | Birth lacks taught ACT preference; plasticity-off blocks D1-shaped preference; strip S + donor changes retrieval; ρ reset preserves `weight_hash`; renamed twin spellings differ and twin does not emit main tokens |
| `G8_trace_purity` | Restore + replay with read-only audit probes matches the uninstrumented trajectory and all six RNG states |

`all_controls_green` iff all eight `ok=true`. Any red **refuses FULLDEV.R7** and authorizes diagnosis → isolated v27, not a rescore.

## G1 — forbidden source (frozen)

Search those two neural files only. Hits anywhere in the file count, including comments.

**Identifiers (substring):**

- `phrase_program`
- `known_chunks`
- `phrase_target`
- `expected_length`
- `expected_len`
- `stored_length`
- `target_length`

**Regexes:**

- `\b_phrase\b`
- `\bstage\s*==`
- `\bdomain\s*==`
- `STOP` assigned after a copied token queue is exhausted (`phrase_program` covers this if present)

Echoic persistence as a **bias** is not itself a G1 hit. Direct copy of the current observe into an emit program is a G1 hit (`_phrase` / `phrase_program`).

## G2 — cross-modal continuation / refractory

Use opaque motor handles (not English). Familiarize one cue symbol. Collect consecutive observes.

- `after_emit_hold` = P(next op is HOLD | current chosen op is EMIT)
- `after_act_hold` = P(next op is HOLD | current chosen op is ACT)

Pass only if the same continuation/refractory shape appears for ACT sequences, not only emitted phrases.

## G3 — non-echo

Present A. Next body moves toward setpoint iff the emit sequence starts with B (not A); toward harm if it starts with A. Pass only if later A-probes emit B more than they echo A. Echoic persistence may bias, but must not be the answer function.

## G4 — order and length

Teach ordered `[P, Q]` on main. Twin is a renamed lexicon of the same roles.

1. **Renamed:** twin probed with twin `[P', Q']` may emit twin tokens; twin probed with **main** spellings must not emit main's `[P, Q]`.
2. **Reversed:** after learning `[P, Q]`, a `[Q, P]` probe must not copy `[Q, P]` on ≥ 8/20 trials.
3. **Distracted:** after learning `[P, Q]`, a `[P, D, Q]` probe must not copy `[P, D, Q]` on ≥ 8/20 trials.
4. **Unseen length:** after learning length 2, a 5-token probe must not emit length 5 on ≥ 8/20 trials.

Behavior must follow learned evidence, not the current token queue.

## G5 — STOP evidence

Same-length (4) sensory phrases. Life 1: beneficial body iff STOP and `len(emit)==2`. Life 2: beneficial iff STOP and `len(emit)==4`. STOP must follow boundary evidence, not a stored or copied length.

## G6 — neophobia provenance

No token whitelist. Familiarity arises from this life's neural/S experience. Strip by `reset_cortex` (relevant neural experience gone). Donor is a checkpoint from a life that experienced the token.

## G7 — ablations

| Fork | Expected |
|------|----------|
| Birth | `press > harm` fails on 40 probes before teaching |
| Plasticity-off | same D1 teach schedule; `press > harm` still fails |
| Stripped S | organism write deleted; donor write changes retrieval norm |
| ρ reset | `weight_hash` unchanged |
| Renamed twin | symbol/handle spellings differ; twin does not emit main's `emit1` |

## G8 — trace purity

Checkpoint. Run 8 observes, record `(op, token, emit_sequence)` and RNG fingerprints (`birth`, `registry`, `source`, `action`, `permute`, `motor`). Restore. Replay the same 8 observes, reading `age`, `emit_buffer`, `last_action`, `weight_hash` between observes. Trajectories and RNG fingerprints must match.

## Branch

- Any red: freeze this result; do not reveal FULLDEV.R7; diagnose; isolated v27 + new narrow gate.
- All green: freeze FULLDEV.R7 runner/scorers and a **fresh** commitment; then reveal and score D0–D12 once.
