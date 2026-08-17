# TM.0.24.WALLMAP results

Product remains **0.0.004**. `earned_next=false`. `ex0s=null`. `eligible_for_000005=false`.

Frozen runner: `TM.0.24.WALLMAP.RUNNER.V5` on clean `origin/main` before answers.

## Diagnostic outcomes

| Q | Question | Passed |
| --- | --- | --- |
| Q1 | Dense n=64 representability (behavioral probe on DIAG.FIT) | **false** (best FIT probe 0.55 &lt; τ=0.60) |
| Q2 | One-genotype developmental reachability on DIAG.CHECK | **false** (adult mean 0.13, CI lower 0.03) |
| Q3 | ES SNR (SE) and gradient stability | **false** (median SNR≈0.96; cos(g1,g2)≈0.01) |
| Q4 | State-only credit chain | **false** |

### Q4 link detail (state-only)

- ACT → body: **pass** (mid-range body; beneficial physics observable)
- body → adv: **pass** (beneficial adv &gt; 0; harmful lower)
- adv → correct eligibility: **fail** — zero `rho_elig` still yields nonzero `W_act_query` motion (consolidation / slow-weight path)
- credit → credited handle logit: **pass**
- later probe / logits: **pass** (weak; plasticity-on probe 0.25 vs off 0.05)

### Q3 variance components (reported separately)

birth≈0.069; world≈0; teacher≈0. SNR denominator was SE, not a variance sum.

## Decision

**Primary bottleneck:** `Q4_credit_chain` (takes precedence over Q2).

**Next change:** Repair the general credit path in a new architecture candidate; then re-run a newly committed reachability diagnostic. Q2 failure does not independently diagnose maturation/replay. Frozen LINEAGE and WALLMAP results remain historical.

Do not increase n from this package. Not a capability earn. QUAL/EVAL remain sealed.
