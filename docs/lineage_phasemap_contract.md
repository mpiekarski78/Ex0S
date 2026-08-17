# Causal-phase address map — TM.0.24.PHASEMAP

**Lab:** TM.0.24.PHASEMAP  
**Product:** Ex0S **0.0.004**  
**Flags:** `earned_next=false`, `ex0s=null`, `eligible_for_000005=false`  
**Live candidate:** v30.  
**n:** **64**

Not a lineage rescore. Not a capability earn. **Not a neural amendment.** No `cortex.candidate.v31.lock`. The 1,536-scalar eligibility budget stays **closed**. No D5.

Authorized by [`lineage_discrimmap.r2.decision.lock`](lineage_discrimmap.r2.decision.lock) and [`lineage_discrimmap.r2.decision.addendum.lock`](lineage_discrimmap.r2.decision.addendum.lock). The write-geometry branch is **closed**. Do not try another optimizer, competitive rule, or prototype variant.

R2 showed eight-cue teaching states are technically hard-margin feasible but not robustly usable (tiny geometric margin, failed probe transfer, failed perturbation). Hard-margin feasible means a boundary can interpolate captured points. Robust boundary absent means the address supplied to learning is not stable enough to support behavior. D3 failure does not authorize a better online rule. Frozen D4 RBF failure is not a proof that every nonlinear readout is impossible; it does not affect the linear refusal.

## Question

At what exact transition does an eight-cue address stop being robust?

## What this freeze does

It generalizes COLLISIONMAP from two cues to the complete 2/4/8-cue battery. The same organism is captured at six causal phases. There is **no plastic amendment**. The already frozen DISCRIMMAP R2 **D1** hard-margin oracle is the measuring instrument.

## Phases (frozen)

| Phase | State | COLLISIONMAP key |
| --- | --- | --- |
| P0 | Cue/input tick | `cue` |
| P1 | Event-end state | `event_end` |
| P2 | Observable-state tick | `observable` |
| P3 | Immediately before zero-input motor evolution | last sensory / `observable` (`pre_motor`) |
| P4 | After zero-input motor evolution | `motor_last` |
| P5 | Stored `ρ_elig` used by credit | `rho_elig` |

P2 and P3 may be identical in this organism. That is a measurement, not a hidden degree of freedom. Contraction \(C_{P2\to P3}=1\) would mean no extra transition.

## Contraction

\[
C_t=\frac{\|\rho_{t+1}^{A}-\rho_{t+1}^{B}\|}{\|\rho_{t}^{A}-\rho_{t}^{B}\|}
\]

\(C_t<1\) is contraction (separation shrinks). Reported as min and mean over cue pairs with a nonzero denominator. It distinguishes an abrupt destruction at one transition from decay across several.

## Per-phase report (frozen)

Same-cue stability across repeats, worlds, and renamed twins. Minimum between-cue separation. Within-cue versus between-cue distance. Clean and perturbed normalized geometric margin via D1. Teaching-to-live-probe transfer. Contraction from the preceding phase. Balanced two-handle classification at 2, 4, and 8 cues.

D1 is imported from the frozen R2 runner. No new solver. \(\gamma_i=y_i(w^\top x_i+b)/\|w\|\) with \(\|w\|\) excluding intercept. Pass requires correct classification and \(\min\gamma\ge 0.01\) plus perturbation (`σ=0.01`, `n=20`, `≥19/20`). Fit teaching select-tick rows only.

## Decision ladder (disjoint, frozen order)

1. All six phases robust at eight cues → `revisit_discrimmap_apparatus`
2. Teaching states robust at the first failing phase but same-phase probes fail → `state_reinstatement_instability`
3. Robust through P4, lost at P5 → `eligibility_snapshot_mismatch`
4. Robust through P3, lost at P4 → `motor_transition_address_collapse`
5. P0 robust and a later phase fails, including P1 → `post_cue_dynamics_destroy_robust_address`
6. P0 already fails at eight cues → `cue_representation_capacity_absent`

Eligibility traces preserve an earlier informative state until delayed consequence. They cannot reconstruct information after it has already collapsed. A later connection-local trace may be authorized only if PHASEMAP finds a last robust phase that can feed both ACT scoring and credit.

## Refuse

D5; neural candidate; installing D1–D4; another optimizer / competitive rule / prototype variant; opening SCORE; 1,536 eligibility budget; larger n; lineage; QUAL/EVAL; FULLDEV.R7; rewriting historical locks; `earned_next`; 0.0.005; instincts; SFNN.
