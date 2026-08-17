# Memory lifecycle on P1 — TM.0.24.MEMORYLIFECYCLEMAP.R2

**Lab:** TM.0.24.MEMORYLIFECYCLEMAP.R2  
**Product:** Ex0S **0.0.004**  
**Flags:** `earned_next=false`, `ex0s=null`, `eligible_for_000005=false`  
**Live candidate:** v30, default v29 query scoring.  
**n:** **64**

Not a lineage rescore. Not a capability earn. **Not a neural amendment.** No trace install. No `cortex.candidate.v31.lock`. Do not resurrect W1. Do not open SCORE. Do not test another isolated write formula. Do not rewrite the published V1 freeze on `ec317ba`.

Authorized by [`lineage_convergencemap.decision.lock`](lineage_convergencemap.decision.lock), [`lineage_convergencemap.decision.addendum.lock`](lineage_convergencemap.decision.addendum.lock), and [`lineage_memorylifecyclemap.runner.addendum.lock`](lineage_memorylifecyclemap.runner.addendum.lock). Historical V1 runner.lock SHA `28cc70a50de9c9f65d3ea351f8d598dd5274751d4bbd956dff5212e1156fa593` is preserved.

## Why R2

V1 is not a matched L2/L3 contrast if only L3 receives a direct live reversal update. Radius-scaled match noise is an implementation sanity check, not the frozen σ=0.01 robustness test.

## Question

Does a compact error-driven rule plus a fixed eight-slot P1 episode store, with content-addressed replacement on reversal, satisfy a *phased* contract that CONVERGENCEMAP's combined monotonic-retention-and-reversal contract refused — when L1–L3 share one live reversal update and then differ only in store/replay, and when ecological match uses the frozen σ=0.01?

## Arms (frozen)

Runner-only. Exact P1 bridge. Error-driven learner is CONVERGENCEMAP C1 (`η=0.15`, `c_max=1`, error-only). No cue string in the store.

| Arm | Memory lifecycle |
| --- | --- |
| L0 | C1 live repetition, no episode store |
| L1 | One live reversal update, then FIFO store + error-driven replay (no invalidation) |
| L2 | Same live reversal update, then content-addressed replacement of exactly one selected slot + replay of the new evidence |
| L3 | Same live reversal update, then refuse contradictory replacement (store and captured rows unchanged) + replay of the stale episode |
| L4 | Sequential RLS ceiling, no live reversal, no invalidation |

L1–L3 are a matched contrast: the same single live reversal update, then different store/replay policies. L4 does not live-train reversal. If L2 and L3 both pass, `replacement_not_causal` is licensed. If only L3 had received extra live help, that code would be unsupported.

Each episode contains only: one 64-d P1, opaque chosen handle, consequence sign/strength, age/version. No cue string, semantic identity, or direct answer channel. At most \(8\times 64=512\) state scalars plus small metadata. Not a larger neural population. Not an installed trace.

Reversal match is P1 L2, not cue name. **0.05 is the preregistered match radius**, not a same-cue ceiling. Selection is nearest distance with deterministic tie-break `(distance, age, slot_index)`. Ranking ties are not unique winners.

Cells report both match-perturbation modes:

- `bounded_match_sanity`: radius-scaled noise (`min(0.01, 0.05/(2√64))`). Implementation sanity. **Cannot satisfy the lifecycle stability gate.**
- `ecological_match_stability`: frozen σ=0.01. Scientific robustness. Failure on L2 eco/spec is `episode_reinstatement_match_failure`.

The lifecycle stability gate remains ranking perturbation at σ=0.01 (native margin 0.01, ≥19/20).

## Phased success contract (frozen)

Temporary errors during early acquisition are reported and are not automatic permanent failure.

1. **Acquisition:** final eight-cue ranking (correct unique winners) within the frozen exposure budget.
2. **Stability:** retain ranking, native margin 0.01, and ranking perturbation (σ=0.01) across no-update probes and REST.
3. **Plasticity:** adapt after a genuine reversal.
4. **Specificity:** reversing one cue must not erase unrelated cues (4-cue map).
5. **Honesty:** checkpoint errors are recorded (`n_checkpoint_errors`) and do not by themselves fail acquisition.

L0 budget: 16 live cycles. L1–L3: one live write pass into the store, the same live reversal update, then 16 error-driven replay epochs. L4: 16 sequential RLS epochs on captured P1 rows.

## Decision ladder (disjoint, frozen order)

1. L0 passes all phases → `live_repetition_sufficient`
2. L0 fails, L1 passes all phases → `fifo_replay_sufficient`
3. L1 fails reversal; L2 passes all phases; L3 fails reversal → `content_addressed_invalidation_supported`
4. L2 and L3 both pass all phases → `replacement_not_causal`
5. L2 fails with reversal matcher misses or ecological match failure → `episode_reinstatement_match_failure`
6. Only L4 passes all phases → `covariance_memory_ceiling_only`
7. Else → `memory_lifecycle_insufficient`

If L2 passes while L3 fails reversal, that is causal evidence for a fast episodic store plus controlled replay/invalidation under matched live reversal.

## Refuse

Trace install; v31/v32; W1; neural edit; D5; opening SCORE; installing 512/1536; larger n; lineage; QUAL/EVAL; FULLDEV.R7; rewriting historical locks including V1 MEMORYLIFECYCLEMAP; `earned_next`; 0.0.005; instincts; SFNN; another isolated write-formula grid; cue strings in episodes; using bounded match sanity as the stability gate.
