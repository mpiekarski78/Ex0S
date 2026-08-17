# Lineage PLASTICITYMAP contract — TM.0.24.PLASTICITYMAP

**Lab:** TM.0.24.PLASTICITYMAP  
**Subtitle:** Developmental motor-learning decomposition  
**Product under test:** Ex0S **0.0.004**  
**Flags:** `earned_next=false`, `ex0s=null`, `eligible_for_000005=false`  
**Not:** TM.0.25, a lineage rescore, a capability earn, QUAL/EVAL, or an n increase.

Live candidate: [`cortex.candidate.v28.lock`](cortex.candidate.v28.lock). Frozen LINEAGE, WALLMAP, and REACH remain historical. n stays **64**. Do not move τ or δ. Do not edit the neural mechanism in this package.

## Why this package

Consolidation leak is repaired in v28 (zero-eligibility ΔW=0). Sanity and C4/C5/C6 survived. REACH still does not approach L0 (adult 0.12 vs 0.60). WALLMAP Q1 reached 0.55 under a bounded optimizer (representability unresolved, not disproven). Q3 SNR remains historically unusable. Another lineage run is unjustified.

Zero-eligibility ΔW=0 is necessary but proves only one Q4 intervention. Before treating Q4 as fully repaired, complete the v28 credit chain on **fresh** diagnostic worlds.

## Package question

Where does developmental motor learning fail, once the organism is forced to experience both causal alternatives under the ordinary v28 law?

## Diagnostics (one confound at a time)

### D0 — Complete v28 credit chain (fresh worlds)

State-only interventions. Pass only if every link holds:

1. Beneficial and harmful physics produce opposite advantage.
2. Zero eligibility ⇒ no plastic motion, including consolidation.
3. Correct prior eligibility is used (current `ρ_elig` beats previous-tick `ρ`).
4. Wrong-tick eligibility fails (random `ρ` does not produce the correct-elig beneficial projection gain).
5. Credited motor projection moves in the right direction.
6. Later actuator probability **and** sampled behavior change vs a plasticity-off twin.
7. Credited tensors move through the intended consolidation boundary; unused tensors do not.

If D0 fails, do not treat Q4 as fully repaired. Later developmental reds are not independently diagnostic of maturation vs exploration.

### D1 — Readout expressivity

Directly set `W_act_query` so the query ranks either opaque handle. Pass: each handle can be made the unique motor-score winner. Interpretation if fail: readout cannot express the ranking.

### D2 — Forced balanced ACT exposure (decisive)

Start identical checkpoint clones. Externally schedule equal opportunities for every opaque actuator. Let physics produce beneficial/harmful consequences. Allow only the ordinary v28 learning law. Release forced control. Probe autonomous actuator probabilities and sampled behavior.

This is not teaching the answer. It ensures the organism experiences both causal alternatives.

Pass: after release, beneficial motor score > harmful, and sampled beneficial ACT count exceeds harmful on one-symbol probes, and the trained clone beats a frozen twin from the same start.

If D0 and D1 pass and D2 fails: general three-factor actor pathway wall (eligibility ownership, update direction, fast/slow separation, or motor-vector credit).

### D3 — Autonomous exploration

Same worlds, ordinary `live_once`, no forced ACT. If D2 passes and D3 fails: action-selection / exploration wall.

### D4 — REST / consolidation retention

If immediate preference exists after D2, run `rest_epoch` then probe. If pre-REST preference succeeds and post-REST fails: consolidation/retention wall.

### D5 — Renamed siblings

If a deterministic world passes D2, repeat the D2 procedure on a renamed sibling world (fresh birth, new handles). If deterministic passes and the sibling fails: developmental generalization wall. Zero-shot transfer without re-exposure is reported, not a pass gate.

### D6 — Outer search

Scored only if D1–D3 pass. Otherwise cite historical WALLMAP Q3 (median SNR≈0.96). If developmental diagnostics pass and the gradient remains unstable: outer-search wall (structured mutations, lower-dimensional genes, batching, variance reduction).

## Increase n

Only if a **stronger** representability diagnostic later indicates state/readout structure is the bottleneck. This package does not increase n. WALLMAP Q1 remains historical (0.55, not disproven).

## Order of freezes

1. This contract + prereg (commit/push before answers)
2. Runner + tests; ABI/synthetic smoke only
3. `docs/lineage_plasticitymap.runner.lock` on clean `origin/main`
4. Score D0→D5, then D6 if released
5. Decision lock

## Refuse

Neural edits in this package; scorer softening; L0-specific circuitry; QUAL/EVAL reveal; rewriting LINEAGE/WALLMAP/REACH locks; `earned_next`; 0.0.005; moving `τ`/`δ`; increasing n; FULLDEV.R7; treating forced exposure as a capability earn; another lineage run before this decomposition; treating DIAG worlds as held-out QUAL/EVAL.
