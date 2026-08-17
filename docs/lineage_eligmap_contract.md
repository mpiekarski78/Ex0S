# Eligibility-address map — TM.0.24.ELIGMAP

**Lab:** TM.0.24.ELIGMAP  
**Product:** Ex0S **0.0.004**  
**Flags:** `earned_next=false`, `ex0s=null`, `eligible_for_000005=false`  
**Live candidate:** v30 (query scoring). W1 prototype mode remains default-off.  
**n:** **64**

Not a lineage rescore. Not a capability earn. **Not a neural amendment.** No `cortex.candidate.v31.lock`. No `cortex.candidate.v32.lock`.

Authorized by [`lineage_writegeom.decision.lock`](lineage_writegeom.decision.lock) and the preserved addendum [`lineage_writegeom.decision.addendum.lock`](lineage_writegeom.decision.addendum.lock): **`w1_query_margin_insufficient__unit_norm_negative_inert`**. W1 repaired discrete ranking and failed usable separation. Refusing v31, SCORE, and lineage was correct. The two W1 failures have different causes (collinear live address vs unit-norm negative inertness). This package measures addresses, not instincts.

## What this freeze does

It freezes a **runner-only** diagnostic that compares candidate addresses taken from the **same** organism trajectories. It does **not** install a new neural law.

The trace that would later be authorized must eventually feed **both** live ACT scoring and delayed credit. Changing only the plastic write while live scoring still queries collapsed `ρ_elig` would leave the crumb-margin failure in place.

## Addresses (same trajectories)

| Address | Meaning |
| --- | --- |
| E0 | Current collapsed `ρ_elig` control (post motor tick; WRITEGEOM W1 query) |
| E1 | State immediately before the zero-input motor transition (`sensory_trajectory[-1]`); diagnostic upper bound |
| Eλ | Ungated leaky connection-local presynaptic trace across ticks |
| EΔ | Input/recurrent innovation at the motor boundary: E1 − E0; diagnostic-only |

For Eλ, one row per bound actuator, even when the rows initially contain identical values:

\[
e_{h,t+1}=\lambda e_{h,t}+(1-\lambda)\rho_t
\]

Rows are actuator slots, not cue-keyed state. Handle strings may index the runner registry; they must never enter sensory input or trace contents. Update every snapshot in `last_trajectory` (sensory and motor). All bound rows receive the same presynaptic `ρ_t` (ungated).

Frozen λ grid, DEV only: `{0.0, 0.5, 0.9, 0.95, 0.99}`. `λ=1` is refused (frozen trace).

## Negative-write geometry (runner-side, separate)

| Law | Meaning |
| --- | --- |
| N0 | Current unit-normalized prototype (W1) |
| N1 | Signed vector with maximum-norm clipping |
| N2 | Unit direction plus bounded scalar confidence |

N1 is the smallest credible alternative:

\[
w'_h=\mathrm{clipnorm}(w_h+\eta\,\mathrm{adv}\,\hat e,\,c_{\max})
\]

A negative consequence then immediately reduces the learned component; clipping prevents unlimited potentiation. `c_max=1`. `η=η_{\mathrm{act}}=0.15`. `adv==0` → bitwise no update.

N0 scoring is cosine. N1 scoring is the **dot product** with unit \(\hat e\) (magnitude-sensitive). N2 scoring is `confidence * cosine(direction, e)`.

N2: uninitialized row stays zero; positive credit sets direction to \(\hat e\) and raises confidence; negative credit lowers confidence immediately and does not rotate the unit vector through the origin; confidence `0` clears the direction.

Ranking batteries use **N0** so address is not confounded with write geometry. N0/N1/N2 run separately on E0, E1, and Eλ at `λ=0.9`.

## Capacity and protocol

Two, four, and eight cues mapped across a **fixed two-handle** set. Balanced. Both teaching orders. Equal positive MID delta. Delayed credit remains the organism clamp → next observe; runner writes use the **select-tick** address (action-owned), not the credit-observe state.

Evaluate each address on:

- Raw pairwise separation (COLLISIONMAP distinctness: cosine `< 0.99` or L2 `> 0.05`)
- ACT ranking margin (`cosine_margin_min=0.01` for N0; native-score margin `0.01` for N1/N2)
- Frozen perturbation: `σ=0.01`, `n=20`, pass if winner holds on `≥ 19/20`
- Delayed credit (organism `adv` from clamp cycle)
- Rename twin (TWIN domain, independent spellings)
- Actuator permutation (rows keyed by opaque id)
- Event boundary (distinct interaction tokens at probe)
- Distractor observes of unused symbols
- REST survival (ingest rest trajectories into Eλ, then probe)

## Declared budget (not installed)

If a later package adds one eligibility row per actuator:

\[
8\times 64=512
\]

Fast prototype + slow prototype + eligibility = **1536** scalars at `H_max=8`. This package does not allocate that state in `NeuralCortex`.

## Decision ladder (first match)

1. **Eλ robustly passes at eight cues** → authorize bounded connection-local eligibility, with its entire state budget declared.
2. **Trace separates, but local linear scoring fails** → next wall is competitive/discriminative write geometry.
3. **Only E1 passes** → timing is the wall; investigate forming the eligibility trace before the motor transition.
4. **No candidate address passes** → do not install another local learner. The recurrent motor transition is still destroying the usable information.
5. **N1/N2 fixes reversal but Eλ fails** → negative plasticity is repaired, but cue addressing remains unsolved.

Robust Eλ means: required 8-cue/2-handle ranking with frozen margin and perturbation, both orders, DEV worlds, plus REST/distractor/event/permutation/rename survival for that `λ`.

## Refuse

Neural edit this package; installing Eλ/N1/N2 in `NeuralCortex`; declaring v31 or v32; opening WRITEGEOM or ELIGMAP SCORE; instincts; survival/reproduction objectives; SFNN cell classes; larger n; lineage evolution; semantic channels; direct reward; cue-keyed traces; RLS in the organism; QUAL/EVAL; FULLDEV.R7; rewriting historical locks including WRITEGEOM decision; `earned_next`; 0.0.005; moving `τ`/`δ`; Q3; treating 8-cue/8-handle calibration as this package; credit-historical-ρ-only while live scoring still reads E0.
