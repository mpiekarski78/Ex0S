# Lineage fitness and statistics contract — TM.0.24.LINEAGE

**Product:** 0.0.004. `earned_next=false`. `ex0s=null`.  
Scientific floors below are frozen **before** seeing lineage organism behavior. Phase 0B may change sample size and compute only.

## Search vs gates

Search fitness **always** (even when `G_k` fails):

```text
F_search = adult_lower_quartile → robustness → efficiency
```

Lexicographic: compare adult lower-quartile first; ties broken by robustness (IQR / across-birth-world std, lower variance better after the adult floor); then efficiency (fewer ACT/EMIT costs, shorter runtime).

Causal gate for curriculum unlock, qualification and RC eligibility **only** (learned k; not C6):

```text
G_k = adult ≥ τ_k
      AND adult − birth ≥ δ_B
      AND adult − plasticity_off ≥ δ_P
```

C6: historical green required; not part of `G_k` margins.

Larger causal margins **never** add fitness.

Plasticity-on and plasticity-off start from the exact same sampled birth cortex, S, world and teacher seeds.

## Frozen floors (not revisable after organism behavior)

All adult scores `A_k` are normalized to `[0, 1]` per capability using the scorer-native success rate or mean probe hit rate. C4/C5/C6 keep their historical pass rules; the `τ_k` values below are the lineage adult floors mapped onto those same units.

Adult floors `τ_k` (normalized [0,1] probe rate unless noted):

- L0_C4 τ=0.60 — beneficial-handle probe rate after revision (matches 24/40)
- L0_C5 τ=0.60 — paired plasticity benefit probe rate; historical mean Δ ≥ 0.10 still required
- L1 τ=0.60 — association / conflict / retention / ρ-reset composite
- L2 τ=0.60 — external S / retrieval / replay-dependent retention
- L3 τ=0.50 — delayed imitation / self vs other (wall probe)
- L4 τ=0.60 — grounding / revision
- L5 τ=0.50 — boundaries / ordered construction (wall probe)
- L6 τ=0.50 — composition / short two-agent exchange (wall probe)

**C6 is a remain-green constraint, not a G_k improvement target.** Birth is already required to be consequence-neutral. Unlock and RC require historical C6 green (`|nuisance| ≤ 0.15`, unchanged). Do not apply `δ_B` / `δ_P` to C6.

C4/C5 historical controls remain mandatory in every level (unchanged scorers; not edited).

**Minimum causal margins (learned k only: L0_C4, L0_C5, L1–L6):**

- `δ_B = 0.05`
- `δ_P = 0.05`

Chosen from existing C5 `MEAN_DELTA_MIN = 0.10` as a smaller but nonzero causal margin. Not estimated from lineage runs.

**Minimum improvement effect for CI unlock:** `Δ_min = 0.05` (lower bound of the one-sided CI on adult − τ_k must exceed 0, and adult − previous-champion must have lower bound ≥ 0 if a previous champion exists). First unlock of a level: CI lower bound on `A_k` ≥ `τ_k`.

## Newborn viability (hard invalidation)

A genome is invalid (fitness −∞, no DEV credit) if any:

- D0 absence fails (birth already prefers the later target; existing D0 rule)
- NaN/Inf in ρ, weights, or body
- `|rho|_2 > 1e6` (seizure/saturation)
- HOLD rate ≥ 0.98 over the first 50 waking ticks
- a single bound actuator accounts for ≥ 0.95 of ACT ops when ≥ 4 actuators are bound
- action entropy = 0 over the first 50 ticks (no exploration)
- `body_state` leaves `[0, 1]^4` after physics clip should have prevented it (implementation bug)
- birth-answer leakage, scorer-only in observe, tokens in θ, phrase machinery, host-selected replay, body improved for “correct utterance”, historical lock integrity fail

## Hierarchical CI (method frozen; precision n later)

- Comparison unit: antithetic pair
- Clustered sampling units: world and birth (not ticks)
- Ticks are never independent observations
- Method: percentile cluster bootstrap
- `n_boot = 9999`
- One-sided `α = 0.05`
- Resampling seed: `20260817`
- DEV.A/B/C: one preregistered check per panel
- Failed panels are never reused
- Consolidation triplet is unused until prospect freeze

Precision (B, W, and confirmatory counts) is **not** frozen here.

Replication escalation (sizes frozen in Phase 0B, names frozen now):

- base — checkpoint champion
- confirmatory — development-consolidation before prospect
- hard maximum — never exceeded; architecture wall may use confirmatory, never an ad hoc extra sample

If required precision is unaffordable: record underpowered or reduce scope. Do not move `τ_k`, `δ_B`, `δ_P`, or L thresholds.

## DEV panel stream

Public stream seed (not a scientific held-out): see `docs/lineage.prereg.lock` field `dev_stream_seed_hex`.

Every checkpoint consumes the next unused disjoint triplet. One temporarily frozen champion is evaluated on that triplet. Genome X clearing A, Y clearing B, Z clearing C does **not** qualify Z.

Before `lineage_prospect.lock`, the exact prospective genome must clear A+B+C **together** on a separate consolidation triplet at the confirmatory tier, with no training between panels.

None of these support held-out generalization claims. QUAL is the first one-shot generalization test.

## Both-arm QUAL/EVAL

Any matched Arm C vs Arm D comparison requires both exact prospects frozen on `origin/main` before the shared QUAL reveal. Both scored once. Neither retrains. If Arm C is not frozen first, refuse a superiority claim.
