# Lineage architecture contract — TM.0.24.LINEAGE

**Lab:** TM.0.24.LINEAGE  
**Subtitle:** automatic evolution and capacity wall of the n=64 developmental cortex  
**Product under test:** Ex0S **0.0.004**  
**Flags:** `earned_next=false`, `ex0s=null`, `eligible_for_000005=false`  
**Ancestor:** candidate v27 (`docs/cortex.candidate.lock`), immutable  
**Does not edit:** `docs/cortex_architecture_contract.md` (SHA `0470d5f8…`)

This is a machine contract. Narrative results formats are deferred until the engine exists.

## Hypotheses

- **H1 — Evolution.** Automatic lineage selection can produce organisms that learn better than v27 without inherited knowledge or manual neural edits, under frozen bounded conditions.
- **H2 — Substrate/search wall.** Determine the first level the n=64 substrate fails to acquire under the frozen architecture, search algorithm, data and compute budget. Not an absolute impossibility claim.

A negative result at L0, L1 or L2 is a primary result.

**Package question:** How far can automatic population evolution improve the learning of a fixed n=64 symbolic developmental substrate across L0–L6 under frozen bounded conditions, and where does that architecture/search/data/compute stack reach an honest wall?

**Primary positive claim if earned:** automatic selection produced a genome that learned more reliably under frozen bounded conditions.

## Symbolic body

Ex0S is **blind and deaf**. The environment consists exclusively of:

- temporally ordered opaque symbols
- source / provenance boundaries
- observable symbolic state
- physics-generated body state (existing 4-float `body_state`: energy, damage, resource, analog)

Actions are opaque symbolic actuator handles. Strings have no innate meaning.

The cortex must **not** know: action meanings; correct answers; episode purpose; curriculum level; teacher intent; whether an event is a lesson or a test.

Body consequences:

- energy/resource change
- discomfort or stability (distance from inherited/evolved setpoints)
- action cost (internal)
- observable physical state change following ACT, **regardless of whether the scorer considers it useful**
- interruption, fatigue and rest opportunity

The teacher’s response is another observable event. The world must **not** improve `body_state` merely because an utterance was “correct.” The cortex decides whether the consequence advanced its internal condition.

Observe ABI is the v1 cortex key set. Host `homeostatic_delta` remains banned. No new observe keys.

## Four channels

- **Genome G** — persists across biological generations. Developmental rules, not adult solutions.
- **Developmental state E** — age, temporary regulatory state, maturation and current plasticity. **Not inherited in TM.0.24.** Transgenerational epigenetic effects are explicitly deferred.
- **Culture C** — persists in teachers and communities. Never **copied** into the newborn cortex or newborn S. During life, cultural interactions may author evidence into individual S exclusively through the ordinary observable ABI. Only what the organism experiences can enter its S.
- **Individual memory S** — one organism’s lifetime. Empty at birth. Not inherited.

## Two arms

Neural topology is fixed at `n=64`. “Randomize topology” means the **world relation graph**.

- **Arm D (main):** developmental genome (init distributions, connectivity/growth/pruning rules, regional biases, local plasticity, neuromodulatory gains, age schedule, setpoints, exploration/consolidation). Each child samples exact birth connectivity and weights from G using a **fresh life-specific developmental RNG**. Siblings share G; they are not clones.
- **Arm C (control):** dense birth matrices plus the same dynamics scalars. Engineering baseline. Cannot set `eligible_for_000005`. Cannot support the strongest developmental-organism claim.

Matched QUAL/EVAL worlds. Both exact prospects must be frozen on `origin/main` before a shared reveal. Unequal compute forbids superiority claims.

Optional offline expressivity control may fit a small finite behavior with dense optimization. Pass proves representability of that bounded behavior. Failure does not prove impossibility.

## REST / replay (substrate; implemented after this freeze)

Host may: provide a rest opportunity; stop external stimulation; expose ordinary fatigue/body state.

Host may not: choose memories; stream all of S; prioritize scorer-relevant rows; label an experience useful.

Replay uses the same learned retrieval query, bandwidth, latency, row limits and cost as waking life. v1 selection mixture (genome-evolved, organism-computed): recency, cortical similarity, internally recorded prediction surprise, random exploration.

Plasticity-on and plasticity-off begin from the **exact same** sampled birth cortex, S, world and teacher seeds. Only plasticity differs.

## Teacher (culture, not oracle)

Teacher adaptation is a permitted cultural channel. It may depend only on observable interaction history, public cultural convention and teacher-local state. It may **not** depend on scorer output, expected organism response, hidden cortex state, curriculum ID or intended answer.

“Simplify after failure” uses observables only, for example: no response before timeout; repeated incompatible actuator result; inability to complete a shared observable goal; explicit HOLD; contradiction with the teacher’s public convention.

The teacher is deterministic given visible history and teacher RNG.

**Audit:** identical observable histories with different scorer annotations must produce identical teacher behavior.

**Teacher-convention counterfactual:** swap the teacher’s cultural convention; leave world physics and symbol registry unchanged. Learned behavior must follow the experienced convention.

## Curriculum (organism never sees level ID)

C4/C5/C6 stay in every level. Previously unlocked levels never disappear.

- **L0** — motor contingency, neutrality, beneficial/harmful revision (first earn target)
- **L1** — association, conflict, retention, ρ reset
- **L2** — external S, retrieval, replay-dependent retention
- **L3–L6** — capacity-wall probes (delayed imitation → grounding → ordered construction → short two-agent exchange)

L6 is not assumed. A rigorous L1/L2 wall is a successful TM.0.24 result. A wall does not authorize scorer weakening or capability-specific functions.

Unlock and RC use `G_k` plus the fitness-contract CI rules. Search uses `F_search` even when `G_k` fails.

## Data layers

- TRAIN — fresh each generation; ES update. `TM024.LINEAGE.TRAIN.g{N}.`
- DEV — disjoint panel-triplet stream; curriculum/stopping only. `TM024.LINEAGE.DEV.`
- QUAL — sealed; one-shot after both prospects frozen. `TM024.LINEAGE.QUAL.`
- EVAL — separate sealed commitment; one-shot after formal RC. `TM024.LINEAGE.EVAL.`

Not FULLDEV.R7. Not `pair_seeds()`.

## Provenance order

Phase 0A (done) → this Phase A freeze → unscored apparatus → Phase 0B SHA-pinned preflight → `lineage_engine.candidate.lock` equal to those SHAs → scored evolution → champion DEV triplets → consolidation triplet → both-arm `lineage_prospect.lock` on clean `origin/main` → QUAL once → EVAL once.

No capability scoring before the engine-candidate lock. No implementation change after that lock without a new lineage version and a new versioned preflight.

## 0.0.005

Unchanged. Partial L0–L5 or an honest wall stays 0.0.004. Eligibility requires the full Arm D D0–D12 gate on QUAL and EVAL. Product stays 0.0.004 until a separate human stamp. `ex0s` stays null at 0.0.005.
