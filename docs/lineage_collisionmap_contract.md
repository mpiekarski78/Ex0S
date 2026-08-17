# Lineage COLLISIONMAP contract — TM.0.24.COLLISIONMAP

**Lab:** TM.0.24.COLLISIONMAP  
**Subtitle:** Cross-cue collision / interference diagnostic  
**Product:** Ex0S **0.0.004**  
**Flags:** `earned_next=false`, `ex0s=null`, `eligible_for_000005=false`  
**Live candidate:** v29 (immutable this package)

Not a lineage rescore. Not a capability earn. Not a neural amendment. n stays **64**.

Authorized by [`lineage_statemap.decision.lock`](lineage_statemap.decision.lock): v29 can learn and retain one cue–action consequence; teaching and probing access effectively the same ρ; S6 failed when two cues required opposing consequences. Delayed credit, event boundaries, HOLD, consolidation, and teacher/probe mismatch are not the present wall. S10/S11 remain separate wiring defects; they do not explain S6 yet.

## Frozen cells (unused worlds)

| Cell | Question |
| --- | --- |
| C0 | At birth, do cues A and B separate across start / cue / event-end / observable / motor ticks / ρ_elig? |
| C1 | After teaching A only, does the A/B trace still separate, or has the attractor changed? |
| C2 | After teaching A then B (S6 protocol), does B destroy A’s ranking? Does substituting saved ρ restore it? |
| C3 | Over independently named cues, what is the effective rank of ρ, and can a frozen linear discriminator separate A from B? |
| C4 | Can a preregistered closed-form ridge readout on frozen ρ_elig assign opposite handles when v29 sequential teaching cannot? |
| C5 | Do C0–C4 qualitative bits survive independent renaming? |

Trace stages (no neural edit; read `sensory_trajectory` / `last_trajectory`):

1. source/start (`v_start` + source)
2. cue ingestion
3. event end (`v_end`)
4. observable-state tick
5. every zero-input motor tick
6. operation and actuator readout (`W_op`, `W_act_query`) at each stage, including `ρ_elig`

Teaching protocol matches STATEMAP S6: cue A → beneficial ACT; cue B → harmful ACT; one clamp each; mid-range body.

## Decision table (first match)

| Observation | Diagnosis |
| --- | --- |
| A/B separate after cue ingestion, then are not distinct at motor-last / ρ_elig | Recurrent/motor-loop attractor collapse |
| A/B remain distinct at ρ_elig, but teaching B destroys A’s ranking | Sequential plastic-write interference |
| Frozen linear discriminator or effective rank < 2 at ρ_elig | Representation/rank failure |
| Frozen ridge readout assigns opposite handles; v29 sequential teaching does not | Plastic update geometry failure |

## Frozen optimizer (C4)

Closed-form ridge only: \(W = Y X^\top (X X^\top + \lambda I)^{-1}\) with \(\lambda=10^{-2}\). State = `rho_elig`. Targets = bound motor vectors. No iterative fit, no sklearn, no post-hoc λ search.

## Refuse

Neural edit this package; two-timescale state; L0-specific circuitry; QUAL/EVAL reveal; rewriting historical locks; `earned_next`; 0.0.005; moving `τ`/`δ`; increasing n; FULLDEV.R7; another lineage run; Q3; importing reward leakage, semantic outputs, or structural growth; treating clamp as a teaching oracle.
