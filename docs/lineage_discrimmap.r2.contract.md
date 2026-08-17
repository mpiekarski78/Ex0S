# Discriminative-write map R2 — TM.0.24.DISCRIMMAP.R2

**Lab:** TM.0.24.DISCRIMMAP.R2  
**Product:** Ex0S **0.0.004**  
**Flags:** `earned_next=false`, `ex0s=null`, `eligible_for_000005=false`  
**Live candidate:** v30.  
**n:** **64**

Not a lineage rescore. Not a capability earn. **Not a neural amendment.** No `cortex.candidate.v31.lock` / `v32`. The 1,536-scalar eligibility budget stays **closed**.

Historical DISCRIMMAP (`d1_fails_robustly` on `TM024.DISCRIMMAP.DEV.`) is **preserved**. This package exists because that runner left hidden scoring degrees of freedom. Scoring-law changes after DEV require a **new runner lock** and **unused worlds**.

## Unused worlds

`TM024.DISCRIMMAP.R2.DEV.` / `TWIN.`. SCORE `TM024.DISCRIMMAP.R2.SCORE.` is reserved and unopened. Do not reopen historical DISCRIMMAP DEV/TWIN/SCORE.

## What is frozen before DEV

Fit **only** on teaching select-tick rows. Evaluate on live probes, renamed twins, and frozen perturbations (`σ=0.01`, `n=20`, `≥19/20`). Report **train** and **probe** geometric margins separately. A training-only D1 pass in 64 dimensions is interpolation.

Normalized geometric margin uses \(\|w\|\) **excluding the intercept**:

\[
\gamma_i=\frac{y_i\,(w^\top x_i+b)}{\|w\|}
\]

Pass requires correct classification **and** \(\min\gamma\ge 0.01\). Margin alone must not hide a sign error.

### D1

Hard-margin SVM by support-vector subset KKT enumeration. Status is `optimal` or `infeasible`. **No** soft-margin \(C\), **no** automatic fallback to a tuned soft-margin classifier, **no** sklearn. Infeasibility is recorded as solver status, not converted into an ordinary classification fail. NaN, infinity, or any unaccepted status fails the run closed.

### D2

Targets \(y\in\{-1,+1\}\). Each row is L2-unit (no mean-centering, no per-feature std). Affine intercept is a trailing column of ones and is **excluded** from \(\|w\|\). \(\lambda=0.01\).

### D3

Online competitive: chosen \(+\), unchosen \(-\). Frozen: \(\eta=0.15\), **1** epoch, sample order = teaching sequence (**no shuffle**), zero initialization, \(\|w_h\|\le 1\) after every update, updates on **every** teaching row (not error-only), intercept \(b=0\). Both teaching orders remain **separate** cells.

### D4

RBF kernel ridge ceiling only. Frozen \(\gamma=0.5\), \(\lambda=0.01\). Not v-eligible. Success is ceiling evidence, not authorization to install nonlinear machinery.

## Decision ladder (disjoint, frozen order)

1. D1 and D3 both pass robustly → `competitive_local_rule_supported`
2. D1 passes robustly and D3 fails → `linear_boundary_exists_local_rule_fails`
3. D1 does not pass robustly and D4 does → `nonlinear_ceiling_only`
4. D1 training is clean but probe/twin/perturbation fails → `linear_interpolation_only`
5. Otherwise D1 robustly fails → `robust_linear_boundary_absent`

Robust D1/D3/D4 means required 8-cue rank cells **and** 2-cue twins, both orders, sign + \(\gamma\ge 0.01\) + perturbation, solver `optimal`.

## Fail closed

Preregistration or runner hashes mismatch; missing or duplicated cell; probe or perturbation rows in a fit; any SCORE-domain identifier in the DEV payload; solver NaN/infinity/unaccepted status; the same frozen DEV execution requested again.

## Refuse

Neural edit; installing D1–D4 in `NeuralCortex`; v31/v32; opening any SCORE domain; rescored historical DISCRIMMAP worlds; sklearn / post-hoc \(\lambda\) / soft-margin \(C\); 1,536 eligibility budget; instincts; SFNN; larger n; lineage; QUAL/EVAL; FULLDEV.R7; rewriting historical locks; `earned_next`; 0.0.005.
