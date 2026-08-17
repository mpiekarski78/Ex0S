# WRITEGEOM / v31 — TM.0.24.WRITEGEOM

**Lab:** TM.0.24.WRITEGEOM  
**Candidate:** v31 (amendment candidate, not a product earn)  
**Product:** Ex0S **0.0.004**  
**Flags:** `earned_next=false`, `ex0s=null`, `eligible_for_000005=false`  
**n:** **64**

Authorized by [`lineage_motorpersist.decision.lock`](lineage_motorpersist.decision.lock) and [`lineage_motorpersist.reaudit.lock`](lineage_motorpersist.reaudit.lock). Equal positive MID consequences and both teaching orders still produce last-write-wins on a shared `W_act_query` outer product. Next is plastic-write geometry, not instincts and not another recurrent population.

## Honest W1 memory

No second global recurrent/context state. W1 adds bounded, exchangeable **actuator-local plastic state**: one fast and one slow 64-dimensional synaptic row per active actuator.

\[
\mathrm{state\_budget}=2\,H_{\max}\times 64,\qquad H_{\max}=8
\]

State may grow with the number of bound actuators, never with cues or episodes. `bind_actuators` of more than `H_max` unique handles is refused.

Prototype rows are actuator slots aligned with the opaque actuator registry, not cue-keyed state. Handle strings may index the registry; they must never enter sensory input or prototype contents.

## Arms

| Arm | Role | Where |
| --- | --- | --- |
| W0 | v29/v30 outer-product control (`p=0`) | live neural until v31 is declared |
| W1 | exchangeable actuator-local prototype | neural; **only v31-eligible candidate** |
| W2 | sequential covariance-aware RLS | runner-only diagnostic; not in `NeuralCortex`; not v31 |

## Authorized W1 neural law

Change **ACT ranking and ACT credit only**. Keep v29 action-owned delayed credit, `ρ_elig`, `W_op` (ACT versus HOLD), EMIT, opaque `bind_actuators`, v30 persist at `p=0`.

Chosen handle `h` only. \(\hat\rho=\rho_{\mathrm{elig}}/\|\rho_{\mathrm{elig}}\|\) when \(\|\rho_{\mathrm{elig}}\|>\varepsilon\), else no update. \(\varepsilon=10^{-12}\).

\[
z_h=\mathrm{proto}_h+\eta_{\mathrm{act}}\,\mathrm{adv}\,\hat\rho,\qquad
\mathrm{proto}'_h=\begin{cases}0 & \|z_h\|\le\varepsilon\\ z_h/\|z_h\| & \text{otherwise}\end{cases}
\]

- `adv == 0` → bitwise no update.
- Negative credit may repel and eventually flip a prototype. That is intended for ecological reversal.
- Uninitialized prototype scores exactly zero.
- Every initialized row is unit length: W1 has **no confidence representation**.
- Repeated reward tests directional stability and ranking margin, not bounded norm.

After the additive+normalize step, apply the same \(\beta\) blend as other plastic tensors, then **re-unit-normalize both** live and slow. Live ACT uses the post-blend live row.

`W_act_query` is retained only for checkpoint compatibility. Do not update it during v31 ACT credit. Do not let REST or consolidation modify it as a hidden unused actor.

Canonical scoring:

```text
actuator_scores(rho) -> dict[opaque_handle, float]
```

Both `_motor_loop` and `motor_scores` must call that method. Preserve `rng_motor` exchangeable tie-breaking when every score is zero.

## Lifecycle

- New handle → zero prototype.
- Same handle rebound → retain prototype.
- Removed handle → dormant in the registry; not scored while unbound; not deleted.
- Handle permutation → exact row permutation with the handles.
- Cue renaming → no prototype effect beyond changed sensory dynamics.
- Missing prototypes in a v29/v30 checkpoint → deterministic zeros. Never infer from motor-vector contents.

## Capacity battery

W1 must not allocate a new prototype for every cue.

- 2 cues, 2 handles — minimal opposing (required)
- 4 cues, 2 handles, balanced — multiple addresses per actuator (required)
- 8 cues, 2 handles, balanced — fixed-memory cue capacity (required)
- 8 cues, 4 handles, two cues each — actuator scaling (required)
- 8 cues, 8 handles — calibration only, **not sufficient evidence**

Counterbalance cue order, handle order, mappings, and teaching order.

## Reversal

1. **Ecological reversal (required W1 pass):** A→h1 positive, then A→h1 negative, then A→h2 positive.
2. **Positive-only reassignment (diagnostic, not a required pass):** A→h1 positive, then A→h2 positive, no negative evidence for h1.

## Margin (frozen before DEV)

Do not use `margin > 0`.

- Absolute cosine margin floor: `0.01`
- Perturbation: additive Gaussian \(\sigma=0.01\) on unit-normalized probe \(\rho\), 20 draws, seed from domain hash
- Pass only if raw margin \(\ge 0.01\) **and** the winning handle remains winner on \(\ge 19/20\) draws
- Report both raw margin and `perturb_stable`

## W2 guardrails

\(\lambda=0.01\). Sequential updates after each teaching credit. Train only on teaching \(\rho_{\mathrm{elig}}\). Identical streams as W0/W1. Report rank, condition number, weight norm, margin, perturbation sensitivity. Huge-coefficient interpolation of near-duplicate states does not count unless ranking survives the frozen perturbation. Never install RLS in the organism.

## Equal-advantage teaching

Identical positive body delta for every trained handle. Both orders. No opposite-sign sequential ACT as the opposing protocol.

## If W1 fails 2-cue opposing on DEV

Stop. Do not open SCORE. Do not write `cortex.candidate.v31.lock`. Next escalation: compact connection-local eligibility, still n=64 — not instincts, not SFNN.

## After v31, still closed

Even if v31 passes, lineage stays closed until S10/S11 are repaired. Survival/reproduction drives are out of scope. SCORE opens exactly once, only after law, thresholds, state budget, runner, and candidate hash are on `origin/main`. C4–C6 alone are insufficient: rerun A0–A11, C4/C5/C6, birth, REST/checkpoint, fresh STATEMAP, localization, permutation/twins.

## Refuse

Larger n; semantic channels; direct reward; a second global recurrent/context state; cue-keyed prototypes; RLS in the organism; instincts; SFNN GRUs; QUAL/EVAL; FULLDEV.R7; rewriting historical locks; `earned_next`; 0.0.005; moving τ/δ; another lineage run; Q3; tuning thresholds after seeing WRITEGEOM DEV; treating 8-cue/8-handle calibration as sufficient evidence; treating positive-only reassignment as a silent required pass.
