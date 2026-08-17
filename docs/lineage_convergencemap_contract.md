# One-shot vs replay on P1 — TM.0.24.CONVERGENCEMAP

**Lab:** TM.0.24.CONVERGENCEMAP  
**Product:** Ex0S **0.0.004**  
**Flags:** `earned_next=false`, `ex0s=null`, `eligible_for_000005=false`  
**Live candidate:** v30, default v29 query scoring.  
**n:** **64**

Not a lineage rescore. Not a capability earn. **Not a neural amendment.** No trace install. No `cortex.candidate.v31.lock`. Do not resurrect W1. Do not open SCORE.

Authorized by [`lineage_tracebridge.decision.lock`](lineage_tracebridge.decision.lock) and [`lineage_tracebridge.decision.addendum.lock`](lineage_tracebridge.decision.addendum.lock). TRACEBRIDGE first-match **p1_not_usable_by_online_class** is preserved and interpreted as **p1_not_usable_by_frozen_one_pass_online_rules**. B3 was one epoch, fixed order, always-update D3. B4 found a positive-margin P1 separator, so an error-driven perceptron can theoretically converge if examples are revisited.

## Question

Is the eight-cue wall caused by one-shot exposure, or does it require covariance/replay machinery?

## Arms (frozen)

Exact P1 bridge, runner-only. No v29 write-law change.

| Arm | Rule | Exposure |
| --- | --- | --- |
| C0 | Existing one-epoch always-update D3 control | one live pass |
| C1 | Error-only competitive perceptron | live re-exposure at 1/2/4/8/16 cycles |
| C2 | One-pass passive-aggressive margin correction | one live pass |
| C3 | Passive-aggressive | live re-exposure at 2/4/8/16 cycles |
| C4 | Sequential RLS on stored P1 rows | exact replay, 16 epochs, diagnostic ceiling |

C1/C3 also run a stored-P1 replay at 16 epochs (no live re-observe) so reinstatement can be separated from the learning rule.

Passive-aggressive target is the existing **0.01 geometric margin**. The update is the minimal τ on the current example (`w_chosen += τx`, `w_other -= τx`). No learning-rate grid.

Live re-exposure regenerates P1 through the organism each time. Exact replay reuses stored P1 rows.

## Battery (frozen)

Balanced 2/4/8-cue maps, both orders, renamed twins, frozen perturbations, delayed consequences, ecological reversal, REST. Retention is scored after every cue, not only at the end. Native ranking margin 0.01 plus perturbation for the final probe. Intermediate retention requires correct ranking of every cue taught so far.

## Decision ladder (disjoint, frozen order)

1. C2, or C1 at one live cycle, passes → `compact_error_correcting_write_sufficient`
2. C1 or C3 pass only after live repetition (k>1) → `repeated_exposure_suffices`
3. Stored replay of C1/C3 passes and live C1/C3 fail → `state_reinstatement_unstable_across_learning`
4. Only C4 passes → `covariance_aware_memory_required`
5. Nothing below B4 passes → `oracle_separability_not_operationally_reachable`

Eight arbitrary one-shot associations without replay are a stronger requirement than ordinary survival learning. Biology enters here through repeated experience, error-triggered plasticity, replay, and metaplasticity — not instincts.

## Refuse

Trace install; v31/v32; W1; neural edit; D5; opening SCORE; 512/1536 budgets; larger n; lineage; QUAL/EVAL; FULLDEV.R7; rewriting historical locks; `earned_next`; 0.0.005; instincts; SFNN; a learning-rate grid.
