# Affine vs homogeneous acquisition — TM.0.24.AFFINEMAP.R2

**Lab:** TM.0.24.AFFINEMAP.R2  
**Product:** Ex0S **0.0.004**  
**Flags:** `earned_next=false`, `ex0s=null`, `eligible_for_000005=false`  
**Live candidate:** v30, default v29 query scoring.  
**n:** **64**

Not a lineage rescore. Not a capability earn. **Not a neural amendment.** Do not rewrite AFFINEMAP V1 on `4a5183e`. Do not open `TM024.AFFINEMAP.DEV.`. Do not investigate consolidation.

## Why R2

V1's intercept extract is valid: homogeneous D1 already ranks frozen M3 rows at four and eight cues. Representational sufficiency does not require an intercept.

V1's printed 1.996 values are not a normalized geometric margin. Homogeneous γ on unit-row features cannot exceed 1. Those values are a two-actuator score gap — for D1, `2(w·x+b)` from scores `{+s,-s}`. V1 `probe_pairs` then used that gap as `min_probe_margin` and as the 0.01 stability pass statistic.

The required correction is robustness scoring, not representability.

## Common statistic (frozen)

Two actuator rows. Effective classifier:

\[
v=\frac{w_+}{\|w_+\|}-\frac{w_-}{\|w_-\|},\quad c=b_+-b_-,\quad
\gamma=\frac{v^\top x+c}{\|v\|}
\]

Pairwise score gap is \(s_+-s_-\) with \(s_h=\hat w_h^\top x+b_h\). It is recorded and **must not be called γ** and **must not satisfy the 0.01 gate**.

D1 embedding: \(w_+=w\), \(w_-=-w\), \(b_\pm=\pm b/\|w\|\). That preserves D1 ranking sign and matches DISCRIMMAP \(y(w^\top x+b)/\|w\|\). PA embedding: existing rows; \(b_h=0\) on A2; learned actuator-local bias on A3.

Every A0–A3 row records separately: `ranking_ok`, `pairwise_score_gap`, `normalized_geometric_margin`, `perturbation_ok`. Acquire pass is ranking. Stability pass is ranking **and** min γ ≥ 0.01 **and** perturbation.

A high pairwise gap with γ < 0.01 is a valid stability fail. Coverage aborts only if stored `passed` disagrees with that conjunction (`pairwise_score_gap_used_as_geometric_gate` or scoring inconsistency).

The homogeneous |γ| ≤ 1 bound applies only when feature rows are unit ±ϵ, \(c=0\), and \(\|v\|>ϵ\). A zero effective separator is invalid and cannot rank.

## Contrasts (frozen)

- A0 vs A1 differs only by free versus zero intercept.
- A1 vs A2 uses the same homogeneous hypothesis class and this common score statistic.
- A2 vs A3 differs only by learned bias.
- An A3 raw-gap increase cannot count as bias support without geometric-margin and perturbation passage.

## Arms and grid

Same A0–A3 learners as V1. Same P1 store, cue counts, twins, perturbations. No lifecycle changes. No eco/spec. 104 cells. Unused `TM024.AFFINEMAP.R2.DEV.` / `TWIN.`. SCORE reserved. A3 diagnostic only, not an instinct, not authorized.

## Decision ladder (disjoint, frozen order)

1. A0 passes acquire ranking at four and eight cues, and A1 fails at four or eight → `affine_intercept_required`
2. A1 passes acquire ranking at four and eight cues, and A2 fails at four or eight → `online_optimization_failure`
3. A2 passes acquire ranking at four and eight cues while M1 previously failed four-cue acquire ranking → `apparatus_inconsistency`
4. A3 passes ranking, normalized geometric margin 0.01, and perturbation at four- and eight-cue **stability**, and A2 fails that robustness at four or eight → `learned_local_bias_supported`
5. A0 and A1 both fail acquire ranking at four cues, or both fail at eight → `d1_ceiling_reaudit`
6. Else → `affine_map_insufficient`

## Refuse

Rewrite or rescore AFFINEMAP V1; open V1 DEV; call a pairwise gap γ; use that gap as the 0.01 gate; authorize A3; neural edit; v31; W1; SCORE; 512/1536; consolidation; eco/spec; lifecycle changes; `earned_next`; 0.0.005.
