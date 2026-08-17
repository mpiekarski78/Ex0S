# Persistence-gate freeze — TM.0.24.PERSISTGATE

**Lab:** TM.0.24.PERSISTGATE  
**Product:** Ex0S **0.0.004**  
**Flags:** `earned_next=false`, `ex0s=null`, `eligible_for_000005=false`  
**Live candidate:** v29 (immutable this package)

Not a lineage rescore. Not a capability earn. **Not a neural amendment.** n stays **64**.

Authorized by [`lineage_collisionmap.decision.lock`](lineage_collisionmap.decision.lock). COLLISIONMAP is completed and frozen.

## Causal result

The failure is one transition:

**observable-state ρ (A/B distinct) → zero-input motor tick → ρ_elig A ≈ ρ_elig B.**

The representational machinery can encode cue identity. The motor transition destroys that distinction before action selection and credit ownership.

Rank-6 across eight cues does not weaken this. Effective rank is a global statistic; it can coexist with a particular A/B pair collapsing. Closed-form ridge exploits tiny residual differences with an unconstrained batch solution that v29’s sequential local update cannot realistically reproduce.

Changing only which historical state receives credit would be insufficient: if the live motor readout still sees the collapsed state, the organism still cannot express cue-dependent actions.

## External work (guidance only)

- **SFNN** is now directly relevant: functional cell/synapse types and sparse topology helped avoid shared-attractor symmetry.
- **LNDP** connection-local memory is premature: the failure happens before credit is applied.
- **MorphoNAS** reinforces that recurrence and plasticity cannot be optimized independently. It does not justify structural growth.

None of those authorize reward leakage, semantic outputs, larger n, structural growth, or a particular implementation now.

## What this freeze does

It records that **investigating generic state persistence is now justified**.

It does **not** authorize a particular implementation. Residual / leaky / gated recurrence inside the existing 64 units is the **smallest generic class to test first if** a later package authorizes an amendment. That class must not know cue identity, handles, rewards, or task roles.

## Amendment gate (frozen; unused worlds)

If a later package authorizes an amendment, it must show all of the following on unused worlds, with COLLISIONMAP distinctness thresholds (`cos_distinct_max=0.99`, `l2_distinct_min=0.05`):

1. A/B remain measurably distinct at `ρ_elig`.
2. Live motor rankings can become opposing under sequential v29-style teaching (the live readout, not a substituted historical ρ).
3. STATEMAP S0–S12, renamed twin, HOLD integrity, birth controls, and consolidation remain valid.
4. No benefit from larger n, semantic channels, or direct reward.
5. Lineage reopens only after a newly committed deterministic reachability diagnostic passes.

## Refuse

Neural edit this package; choosing residual vs leaky vs gated now; two-timescale as a committed design; credit-historical-ρ-only patches; L0-specific circuitry; QUAL/EVAL reveal; rewriting historical locks; `earned_next`; 0.0.005; moving `τ`/`δ`; increasing n; FULLDEV.R7; another lineage run; Q3; reward leakage; semantic outputs; structural growth.
