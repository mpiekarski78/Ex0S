# CORTEX v34 architecture amendment

Authorized by [`lineage_competitive.decision.addendum.lock`](lineage_competitive.decision.addendum.lock).

v33 competitive plasticity: geometry-only tie-mean rival depression at P1 preserved v32 triggers; eight-cue acquire remains **7/8** (`competitive_core_acquire_fail`). Analytic interpretation: the dominant replay-all component performs 16 uniform REST passes / 128 row updates and repeatedly adds the same centroid term; secondary low-margin strengthening did not change the observed result. This amendment replaces **uniform replay** with **bounded prediction-error-gated competitive rehearsal** while preserving representation, eight-slot store, v33 competitive geometry, and all hyperparameters.

Not another solver MAP. Not low-margin awake learning alone. Not unbounded organism diagnostic memory.

Preserve the entire v33 organism except replay triggers:

1. **Store and geometry frozen:** eight content-addressed P1 episodes; contradictory replacement; match radius 0.05; v33 positive `(m_h − m_R) ⊗ P̂₁` and negative `m_h`-only vectors; tie-mean rival R within `TIE_EPS`.
2. **Awake live credit frozen:** on ACT credit, `_episode_write` then immediate update only when `_act_ranking_error(p1, handle, adv)`.
3. **Violation predicate** `_episode_rehearsal_violation(p1, handle, adv)` for stored-row rehearsal:
   - **Positive (`adv > 0`):** unique winner ≠ handle, winner is None (tie), or normalized geometric margin `γ(handle) < 0.01`.
   - **Negative (`adv < 0`):** winner == handle (harmful still wins) or winner is None.
   - **Single actuator:** when only one handle is bound, positive rows require `γ ≥ 0.01`; negative rows require winner ≠ handle.
4. **Awake rehearsal burst** after each episode write while not RESTing:
   - Up to **16 passes** over all valid stored episodes (frozen `EPISODE_REPLAY_EPOCHS` budget per burst).
   - Each pass: count violations → apply competitive update **only on violating rows** → recount.
   - Stop early when zero violations remain; else exhaust budget.
   - **Order:** episode write → immediate ranking-error update (if any) → rehearsal burst.
   - After the immediate awake update, **do not** re-update the episode just written if it already has safe margin.
   - Clip during burst; no slow mix until REST block.
5. **REST rehearsal** replaces unconditional replay-all and separate strengthen:
   - 16 epochs of the **same gated pass** (violation-only updates).
   - One slow β mix into `W_slow` after the 16 epochs.
   - Record violations before slow mix and after slow mix separately.
6. **Bounded diagnostics:** each call returns a fixed-size summary; runner aggregates externally. No lifetime-growing diagnostic store on the organism (512-scalar budget).
7. Preserve n=64, η=0.15, query scoring, erratum tie band, missing-`rho_p1` fallback, checkpoint counters.
8. Do not edit `cortex_develop_scorers.py`. Do not rewrite historical locks. Do not auto-write `cortex.candidate.v34.lock`. Do not stamp `earned_next` or 0.0.005.

**If v34 fails:** stop modifying linear write rules. Next jump is complementary learning — episodic P1 store participates in ACT scoring as a fast hippocampal path while `W_act_query` remains the slow cortical path.

Narrow claim: prediction-error-gated competitive rehearsal on unused `TM027.GATEDREHEARSAL.*` worlds. Score on 54-cell battery with consolidation checkpoints. Lineage stays closed even if battery passes until separate candidate freeze.
