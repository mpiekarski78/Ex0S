# MOTORPERSIST / v30 — TM.0.24.MOTORPERSIST

**Lab:** TM.0.24.MOTORPERSIST  
**Candidate:** v30 (amendment candidate, not a product earn)  
**Product:** Ex0S **0.0.004**  
**Flags:** `earned_next=false`, `ex0s=null`, `eligible_for_000005=false`  
**n:** **64**

Authorized by [`lineage_collisionmap.decision.lock`](lineage_collisionmap.decision.lock) and [`lineage_persistgate.prereg.lock`](lineage_persistgate.prereg.lock). COLLISIONMAP localized the wall to the zero-input motor tick. PERSISTGATE justified investigating generic persistence and forbade a particular implementation until this freeze.

## Authorized neural law

Change **only** the zero-input motor transition (`_sensory_tick` with `record_sensory=False`):

\[
\tilde\rho_{t+1}=f(W_{\mathrm{rec}}\rho_t),\qquad
\rho_{t+1}=p\rho_t+(1-p)\tilde\rho_{t+1}
\]

`p` is one scalar persistence coefficient. `f` is the existing tanh recurrent map (including body/`W_in` on the zero-symbol motor tick). No extra L2 renormalization: `p=0` is exactly v29. Convex combination of tanh states stays in \([-1,1]\).

Keep unchanged: n=64; v29 action-owned credit; actual post-motor state remains `ρ_elig`; operation and actuator readouts; consolidation; opaque handles; sensory sequence; no semantic cue/action channels; no additional memory vector.

Do **not** credit a saved earlier cue state while the live actuator still reads a collapsed `ρ`.

## Development `p` grid (unused DEV worlds)

Preregistered: `p ∈ {0.0, 0.25, 0.5, 0.75, 0.9, 0.95}`.

Choose the **smallest** `p` that simultaneously:

1. preserves meaningful A/B separation at `ρ_elig` (COLLISIONMAP thresholds: cosine `< 0.99` or L2 `> 0.05`);
2. permits opposing sequential teaching on those DEV worlds;
3. retains S0 single-cue learning;
4. does not make motor output constant or disable the motor transformation (post-motor step L2 at least 25% of the `p=0` step).

Then freeze that `p` and the implementation **before** scored worlds. Do not tune on scored cells.

## Scored gates (fresh unused worlds)

Behavioral correctness is decisive. Cosine/L2 are explanatory.

| Gate | Question |
| --- | --- |
| P0 state preservation | A/B separation survives the motor tick across renamed cue pairs |
| P1 behavioral rescue | Teach A→handle 1 then B→handle 2; live probes produce opposing rankings |
| P2 order control | B-first then A-second also succeeds |
| P3 revision | Reversing one cue changes only that cue’s learned consequence |
| P4 STATEMAP regression | S0–S12 conclusions remain valid on fresh worlds |
| P5 integrity | HOLD, REST, consolidation, reset, distractor, rename, no-teaching, birth |
| P6 no innate solution | Persistence must not produce correct cue-specific behavior before observable consequences |

## If scalar persistence fails

- Identity still collapses for every usable `p` → generic context/motor partition inside 64 units.
- Identity survives, opposing learning still fails → plastic-write geometry / compact connection-local state.
- High `p` preserves identity but destroys motor computation → learned gate or functional cell classes (SFNN, useful part only).
- It succeeds → freeze v30, then **deterministic reachability** (lineage still closed).

## After v30, still closed

Even if v30 passes, lineage stays closed until S10/S11 are repaired: `lineage_params` in the live factory; age stages affect runtime; every evolved gene has a reader or is removed; deterministic reachability passes; then Q3 SNR. Not this package.

## Refuse

Larger n; semantic channels; direct reward; extra memory vector; crediting historical ρ while live readout stays collapsed; QUAL/EVAL; FULLDEV.R7; rewriting historical locks; `earned_next`; 0.0.005; moving τ/δ; another lineage run; Q3; tuning `p` on scored cells.
