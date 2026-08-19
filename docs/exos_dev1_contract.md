# EX0S-DEV1 Contract

**Program:** EX0S-DEV1  
**Product under test:** Ex0S 0.0.004 (no automatic increment)  
**Architecture lock:** [`docs/exos_dev1.architecture.lock`](exos_dev1.architecture.lock)

---

## Mission

Build a newborn, text-native artificial organism that learns a small artificial language entirely through interaction, remembers facts across resets, answers unseen compositions, revises beliefs, asks for clarification when evidence is ambiguous, and later improves across generations.

The organism may inherit generic neural organization and plasticity rules. It must not inherit vocabulary, facts, language solutions, task-specific routes, or expected answers.

This is not an LLM program. There is no pretrained language model, Transformer, external corpus, planted semantics, hidden answer channel, or fine-tuning on evaluation dialogues. Cortical weights must change during the organism's lifetime: online synaptic plasticity is part of the organism, not LLM retraining.

---

## Governing architecture decision

Rejected assumption: an immutable external record store plus a frozen cortical decoder is not sufficient as the organism's complete memory system.

| State | Owner | Lifetime | May contain | Must not contain |
|-------|-------|----------|-------------|-----------------|
| Genome G | Lineage | Inherited | Module topology, birth-init law, motor/sensory organization, local plasticity rules, developmental timing (age/internal signals only), replay policy | Vocabulary, fixture facts, expected answers, task-specific symbol meanings |
| Slow cortex W | Organism | Learned during life | Concepts, skills, action policy, reusable relational and language structure | Exact autobiographical records supplied by runner |
| Fast memory H | Organism | Fast-changing during life | Episodes, one-shot facts, temporal/source evidence, organism-generated indices | Programmer keys, logical slots, answer handles, task types |
| Working state ρ | Organism | Transient | Current context, active inference, retrieved content | Persistent facts |
| Provenance log S_log | Audit boundary | Immutable | Raw experienced events, timestamps, organism-owned receipts | Retrieval addresses, corrective actions, direct behavioral routing |

S_log is not the hippocampus. H is. S_log is append-only and has **no outgoing behavioral connection** to any other component.

---

## Within-life learning law

### Allowed signals

- Sensory events and their temporal order
- The organism's chosen motor action and efference copy
- Teacher-demonstrated action through the same motor-observation channel
- Scalar reward/advantage as a gate, never as an answer identifier
- Internally generated prediction error, novelty, conflict, and eligibility traces
- Replay sampled by the organism from H

### Forbidden signals

- Expected action handles supplied directly to a weight update
- Cue IDs, logical slots, task-family labels, or fixture metadata
- Runner-generated keys, queries, stored values, neural targets, or retrieval addresses
- Future probes or validation worlds in learning
- An offline oracle solution installed into the organism
- An LLM, external dictionary, corpus, or planted semantic vector

### Gradient law

**No gradient-based update is applied to W, H, or ρ during an organism's evaluated lifetime.** Within-life learning uses only the frozen local plasticity law: pre/post activity, eligibility, neuromodulation, prediction errors, and organism-sampled replay.

The research optimizer may differentiate through completed training-life trajectories, but it may update only inherited G between lives. Validation and confirmation lives never contribute gradients.

### Developmental schedules

Developmental schedules encoded in G are driven by generic age/internal signals only. They must never reference named curriculum stage labels.

---

## Stage definitions

### Stage A — Grounded cortical plasticity

One continuous newborn life learns four motor roles from interaction and consequences.

**H write/read is disabled throughout Stage A** so fast memory cannot solve what is described as cortical grounding.

Required:
- Renamed symbols and unseen contexts pass
- Action identity remains separable across later development
- Ordinary learned actions retained
- Reward-off, feedback-off, permuted-feedback controls behave causally
- No persistent-memory claim made yet

### Stage B — Fast memory ownership

**Slow cortical plasticity and consolidation are disabled during one-shot fact acquisition.** Probes occur before any cortical transfer. This prevents slow cortex from solving the one-shot probe (reproducing TM045 `memory_not_necessary`).

Required:
- No-H arm fails the one-shot probes; H arm passes — with slow cortex disabled in both
- One-shot facts survive `EpisodeReset`
- Early H wipe removes the facts
- `FullCheckpoint` + restore works without harness re-keying
- `HippocampalGraft` between matched donor twins redirects the facts  
  (Donor twins: clones of the same pre-teaching `FullCheckpoint`, diverged by different fact experiences before graft)
- Renamed cues and new worlds pass
- Organism generates every retrieval address

Stage C then deliberately re-enables consolidation.

### Stage C — Semantic consolidation and organism-owned addressing

Required:
- Repeated structure transfers into cortical weights W
- Cortical behavior survives H lesion for the consolidated regularity
- Unrelated episodic details remain H-dependent
- Learned addressing rejects unfamiliar cues without programmer keys
- Replay/consolidation ablation isolates the mechanism actually claimed
- Later learning remains possible

### Stage D — Micro-language lifetime

One uninterrupted life must:
- Learn grounded meanings
- Learn and retain facts across `EpisodeReset`
- Revise a belief without erasing unrelated knowledge
- Answer unseen two-step compositions
- Retain older meanings after further development
- Emit ASK/HOLD under ambiguous or conflicting evidence (via motor channel, not hardcoded op)
- Accept clarification and answer correctly
- Continue learning afterward
- Preserve Stage B wipe/donor causality at appropriate memory timescale

### Stage E — Organism evolution

Unlocks only after Stage D confirmation.

Required:
- Descendants inherit G only; W, H, ρ, S_log begin empty/newborn
- Fitness measured over multiple complete lives
- Held-out worlds never participate in parent selection
- Success: faster or more reliable development, not inherited answers

Stage E biological-evolution claims require the evolutionary path. Meta-gradient discovery is a separate comparison arm and is not evidence of organism evolution.

---

## Causal decision ladder (ordered)

1. `setup_precondition_fail`
2. `semantic_leakage`
3. `feedback_not_causal`
4. `memory_not_necessary`
5. `address_not_organism_owned`
6. `checkpoint_or_reset_fail`
7. `grounding_fail`
8. `fast_memory_fail`
9. `consolidation_fail`
10. `composition_fail`
11. `revision_fail`
12. `clarification_fail`
13. `continued_learning_fail`
14. `integrated_development_pass`

Controls are required failures only when the corresponding causal claim requires them. Observations must not be promoted into causal gates after results are seen. Geometry, RSA, sparsity, and neural hashes are telemetry — they cannot overrule behavioral and intervention gates.

---

## Candidate and product policy

- A discovery pass is not a cortex candidate.
- A stage confirmation freezes a research candidate for the next stage only.
- No `cortex.candidate.v41.lock` before Stage D passes an untouched confirmation and a separate architectural review.
- No `0.0.005` or live-pointer update before an explicit product-earn decision.
- No historical lock is rewritten to make the new lineage appear continuous.
- Each stage is preregistered immediately before execution. Stage preregistration is never deferred past execution.

---

## Definition of progress

The program makes progress when it either:
- advances an integrated behavioral and causal stage on untouched worlds; or
- exhausts a structurally varied, preregistered search budget and localizes the failure to a major subsystem with a predetermined redirect.

A new diagnostic, an in-sample decoder, an oracle W*, a geometry improvement, or more passing unit tests does not count as organism progress.

---

## Refuse list

`no_auto_stamp`, `no_v41_before_stage_d_confirmation_and_review`, `no_W_star`, `no_TM063_diagnostic`, `no_rewrite_historical`, `no_planted_semantics`, `no_gradient_updates_during_life`, `no_operand_field_unless_organism_learned`, `no_stage_prereg_deferred_past_execution`

---

## Redirect rules

| Pattern | Redirect |
|---------|---------|
| Scale-only wins | Expand capacity before adding mechanisms |
| Modular small beats monolithic large | Architecture is causal, not size |
| Episodic memory works; semantic transfer fails | Change cortical consolidation/objectives |
| Semantic transfer works; composition fails | Change relational representation |
| Behavior passes; wipe/donor fails | Reject candidate as non-memory-controlled |
| All sizes and plasticity families fail at grounding | Reconsider sensorimotor/reward interface |
| Only oracle-trained or future-constrained solutions pass | Reject; no organism mechanism earned |

---

## Research basis

- Inherited organization as compressed developmental machinery: Zador, "A critique of pure learning".
- Fast episodic learning plus slow extraction of structured cortical knowledge: Kumaran, Hassabis & McClelland, complementary learning systems.
- Separate hippocampal pathways for episodic detail and statistical regularity: Schapiro et al.
- Factorized structural/content representations and relational generalization: Tolman–Eichenbaum Machine.
- Stable behavior despite representational drift requires a compatible readout: Rule et al.
- Generic plasticity rules can be optimized across tasks while learning novel content inside a lifetime: Miconi et al., differentiable plasticity.
- Scale must be interpreted together with organization: FlyWire connectome ~139,000 neurons, ~54.5M synapses, many cell types.
