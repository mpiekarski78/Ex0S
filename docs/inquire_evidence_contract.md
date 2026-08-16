# Inquire-evidence contract — TM.0.19.INQUIRE Phase B

**Lab:** TM.0.19.INQUIRE.MECH

**Product under test:** Ex0S **0.0.004**

**Flags:** `earned_next=false`, `ex0s=null`

**Prereg:** [`inquire_mech.prereg.lock`](inquire_mech.prereg.lock)

## Claim

> An opt-in recipe may derive competing hypotheses from factorized S evidence, score one-step epistemic partition value then locked cost, and return `plan_inquiry` → ANSWER | PROBE_ATOMS | SYMBOLIC_ACTION | HOLD without calling the teacher. Host-executed consequences enter ordinary grounding channels; `experience_inquire` stores plans/traces only. Budget 8; scored depth ≤ 4; inquiry metadata alone never substitutes for world evidence.

## Epistemic scope

Bounded **one-step** partition scoring from S; replan after each host observation. Not route LOOKAHEAD / multi-step search. I8 = repeated myopic inquiry.

## Stepwise ABI

`plan_inquiry({context_atoms, input_symbols})` → status in {ANSWER, PROBE_ATOMS, SYMBOLIC_ACTION, HOLD}.

Host executes probes and writes consequences via frozen `observe_symbol_ground` (and related). Organism never callbacks the teacher.

## Dual memory

| Channel | Role |
|---------|------|
| `experience_inquire` | plans / traces only |
| grounding / events / sequence | ordinary world consequences |

Strip consequences → ambiguity returns. Strip inquire plans, keep consequences → conclusion remains.

## Value

`value(p) = |H| - size(largest outcome bucket)`; then minimum locked cost; unique winner else HOLD.

## Expression

`PROBE_ATOMS` / `SYMBOLIC_ACTION` require `emit_sequence(context_atoms, probe_atoms)` to uniquely render exactly those atoms; otherwise HOLD (`cannot_express`). No parallel English generator.

## Budget

Hard budget **8** (scored depth ≤ 4). Exhaustion → HOLD.

## Pin order

Unscored smoke → `inquire.candidate.lock` → cells → I-life → capacity → wall (wall prereg already frozen). Preserve `inquire.candidate.v1.lock` if audit rewrites the agent after score.
