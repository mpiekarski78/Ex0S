# TM.0.20.RELIABILITY results: evidence-source calibration

**Ex0S under test:** **0.0.004** (not a new stamp)  
**Lab:** TM.0.20.RELIABILITY  
**Date:** 16 August 2026  
**ok:** `true`  
**life_last_stage_clear:** `R12`  
**first_fail (life):** `null`  
**Wall first_fail:** `W_competent_liar` / `diagnostic_fail` → next-primitive hint **honesty**

Locks: [`reliability_baseline.lock`](reliability_baseline.lock) · [`reliability.candidate.lock`](reliability.candidate.lock) · [`reliability.candidate.v1.lock`](reliability.candidate.v1.lock) · [`reliability_mech.lock`](reliability_mech.lock) · [`reliability.lock`](reliability.lock) · [`reliability_wall.lock`](reliability_wall.lock)

`earned_next`: **false** — no Ex0S 0.0.005 / 1.0. Product stamp remains **0.0.004**.

## Bounded claim

> Context-conditioned predictive reliability learned from independently verified outcomes.

Expanded: Ex0S derived a context-conditioned **source_evidence_margin** (predictive accuracy) from organism-compared claim↔independent-outcome evidence, used it to weight later non-duplicated testimony conflicts, retained uncertainty without sufficient verification, and revised the margin as append-only evidence accumulated—without claiming honesty, trustworthiness, access, independence, or intent.

## What cleared

| Phase | Result |
|-------|--------|
| A baseline (`make_inquire`, reliability off) | conflict → HOLD; no spontaneous weighting |
| B unit cells | **6/6** (incl. real donor / strip-calibration isolation) |
| C life R0–R12 + twin | clear through **R12** |
| Capacity lanes | ok (N-way sources; honest verified/age rungs) |
| Wall scored scripts | pass (`W_indistinguishable_cause`, `W_circular_testimony`); `W_competent_liar` = executed **diagnostic_fail** |
| Wall diagnostic first fail | `W_competent_liar` → **honesty** (`not_run` probes are not diagnostics) |

## Explicit non-claims

- Not a `trust_score`
- Does not implement honesty / access / independence / intent models
- Wall probes those six social-cognition dimensions diagnostically only
- Host never supplies `confirm|contradict`
- Default `make_inquire` / INQUIRE locks unchanged

## Essential breakthrough

> Ex0S did not treat testimony as truth; she kept opaque channel identity and factorized claims, compared claims to independent world observations herself, earned context-conditioned evidence weights in S, used them on non-duplicated later conflicts, and withdrew when verification was stripped—without a host “was the teacher right?” field.

## Audit notes (apparatus)

Post-freeze audit fixes (scientific claim unchanged):

1. **Live supersede persistence** — `live=0` now `store.write`s so TagStore reload cannot resurrect replaced claims.
2. **`testimony_derived` anti-circular** — excluded from `_grounding_support` / inquire hypotheses.
3. **Failure pairing** — failure of a different `paired` no longer contradicts an unrelated claim.
4. **Event-token case** — correlation id normalized to lowercase.
5. **U5 donor / strip-calibration forks** — real isolation (no tautology); keep-testimony/strip-reliability covered.
6. **Capacity honesty** — sources rung uses N live claimants; fake 1k/10k metric rungs removed; metric_only never forces `ok=True`.
7. **Wall first_fail** — `W_competent_liar` is an executed diagnostic (`diagnostic_fail`); `not_run` probes are not diagnostics.

## Next

Multi-factor source model (honesty / access / independence / intent / stability) remains open — named by wall `first_fail_wall`, not earned here.
