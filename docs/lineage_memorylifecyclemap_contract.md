# Memory lifecycle on P1 — TM.0.24.MEMORYLIFECYCLEMAP

**Lab:** TM.0.24.MEMORYLIFECYCLEMAP  
**Product:** Ex0S **0.0.004**  
**Flags:** `earned_next=false`, `ex0s=null`, `eligible_for_000005=false`  
**Live candidate:** v30, default v29 query scoring.  
**n:** **64**

Not a lineage rescore. Not a capability earn. **Not a neural amendment.** No trace install. No `cortex.candidate.v31.lock`. Do not resurrect W1. Do not open SCORE. Do not test another isolated write formula.

Authorized by [`lineage_convergencemap.decision.lock`](lineage_convergencemap.decision.lock) and [`lineage_convergencemap.decision.addendum.lock`](lineage_convergencemap.decision.addendum.lock). Historical first-match **oracle_separability_not_operationally_reachable** is preserved and interpreted as **oracle_separability_not_operationally_reachable_under_frozen_monotonic_retention_and_reversal_contract**.

C1 exact replay ranked all eight cues. What failed was demanding correct ranking after every intervening update *and* later replacing obsolete associations. That is stability–plasticity. The remaining wall is memory lifecycle: acquire → revisit → stabilize → invalidate obsolete evidence. Not representation, reward, motor computation, or raw capacity.

## Question

Does a compact error-driven rule plus a fixed eight-slot P1 episode store, with content-addressed replacement on reversal, satisfy a *phased* contract that CONVERGENCEMAP's combined monotonic-retention-and-reversal contract refused?

## Arms (frozen)

Runner-only. Exact P1 bridge. Error-driven learner is CONVERGENCEMAP C1 (`η=0.15`, `c_max=1`, error-only). No cue string in the store.

| Arm | Memory lifecycle |
| --- | --- |
| L0 | C1 live repetition, no episode store |
| L1 | Fixed eight-slot P1 episode store + error-driven replay (FIFO, no invalidation) |
| L2 | Same store with content-addressed replacement on contradictory evidence |
| L3 | Same store; on a content match with contradictory evidence, retain the old episode and refuse the replacement. Live reversal still trains; replay keeps presenting the stale episode |
| L4 | Sequential RLS ceiling, no invalidation |

Each episode contains only: one 64-d P1, opaque chosen handle, consequence sign/strength, age/version. No cue string, semantic identity, or direct answer channel. At most \(8\times 64=512\) state scalars plus small metadata. Not a larger neural population. Not an installed trace.

Reversal match is P1 L2, not cue name. **0.05 is the preregistered match radius**, not a same-cue ceiling. PHASEMAP P1 8-cue rank cells overlap: `between_l2_min≤within_l2_max`. The radius stays below `between_l2_min`, so it may avoid false matches while missing genuine same-cue reversals. Selection is nearest distance with deterministic tie-break `(distance, age, slot_index)`. Cells record same-cue recall, cross-cue false-match rate, reversal-match recall, twin and perturbation match stability, and `no_match` / `unique_match` / `multiple_match` counts. If L2 fails with reversal matcher misses, that is not scored as a replacement failure.

## Phased success contract (frozen)

Temporary errors during early acquisition are reported and are not automatic permanent failure.

1. **Acquisition:** final eight-cue ranking (correct winners) within the frozen exposure budget.
2. **Stability:** retain ranking, native margin 0.01, and perturbation across no-update probes and REST.
3. **Plasticity:** adapt after a genuine reversal.
4. **Specificity:** reversing one cue must not erase unrelated cues (4-cue map).
5. **Honesty:** checkpoint errors are recorded (`n_checkpoint_errors`) and do not by themselves fail acquisition.

L0 budget: 16 live cycles. L1–L3: one live write pass into the store, then 16 error-driven replay epochs. L4: 16 sequential RLS epochs on captured P1 rows.

## Decision ladder (disjoint, frozen order)

1. L0 passes all phases → `live_repetition_sufficient`
2. L0 fails, L1 passes all phases → `fifo_replay_sufficient`
3. L1 fails reversal; L2 passes all phases; L3 fails reversal → `content_addressed_invalidation_supported`
4. L2 and L3 both pass all phases → `replacement_not_causal`
5. L2 fails with reversal matcher misses → `episode_reinstatement_match_failure`
6. Only L4 passes all phases → `covariance_memory_ceiling_only`
7. Else → `memory_lifecycle_insufficient`

If L2 passes while L3 fails reversal, that is causal evidence for a fast episodic store plus controlled replay/invalidation.

## Refuse

Trace install; v31/v32; W1; neural edit; D5; opening SCORE; installing 512/1536; larger n; lineage; QUAL/EVAL; FULLDEV.R7; rewriting historical locks; `earned_next`; 0.0.005; instincts; SFNN; another isolated write-formula grid; cue strings in episodes.
