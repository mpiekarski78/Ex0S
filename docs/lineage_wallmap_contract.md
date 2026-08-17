# Lineage WALLMAP contract — TM.0.24.WALLMAP

**Lab:** TM.0.24.WALLMAP  
**Subtitle:** L0 wall decomposition  
**Product under test:** Ex0S **0.0.004**  
**Flags:** `earned_next=false`, `ex0s=null`, `eligible_for_000005=false`  
**Not:** TM.0.25, a new lineage version, a capability earn, or QUAL/EVAL.

Ancestor wall: TM.0.24.LINEAGE scored L0 wall (champion adult means 0.09–0.13 vs τ=0.60). Frozen LINEAGE engine candidate remains immutable. Frozen LINEAGE and WALLMAP results remain historical.

Do **not** edit: `docs/cortex_architecture_contract.md`, `experiments/cortex_develop_scorers.py`, `three_memory/neural_cortex.py` during diagnosis, QUAL/EVAL secrets, or lineage scientific floors.

## Package question

Where does the bounded L0 failure decompose among representability, developmental reachability, ES signal-to-noise, and the causal credit path?

## Four diagnostic questions

### Q1 — Representability

Can any dense n=64 adult cortex express bounded L0?

Pass: after frozen dense adult-weight optimization on `Q1.DIAG.FIT`, behavioral `probe_beneficial ≥ 0.60` on ≥ 3/4 fitted worlds.  
Fail: unresolved representation/optimizer wall (not proof of impossibility).  
`Q1.DIAG.TRANSFER` is secondary zero-shot reporting only. A world used for optimization is not held out.

### Q2 — Developmental reachability

Can **one** Arm D genotype learn L0 across renamed worlds without inheriting Q1 adult weights?

Pass on `Q2.DIAG.CHECK` with the same genotype: birth and plasticity-off below τ=0.60; adult ≥ 0.60; `δ_B=0.05` and `δ_P=0.05`; ordinary teaching and credit; fresh sibling RNGs; CI lower bound ≥ τ and `G_k`.  
A favorable birth is not reachability. No per-evaluation-world genotype.  
If Q4 is broken, a Q2 fail does **not** independently diagnose maturation/replay.

### Q3 — Search signal

Can ES distinguish beneficial mutations from replicate noise?

```text
Δ_ir = F(θ + σ ε_i ; z_r) − F(θ − σ ε_i ; z_r)
SNR_i = |mean_r(Δ_ir)| / SE_r(Δ_ir)
g_b = (1 / (2 P σ)) Σ_i Δ_i ε_i
```

Denominator is the **standard error** of `Δ_ir`, not a sum of variances. Report birth/world/teacher variance components separately. Do not use raw mean signed mutation effect.

Pass (all): median `SNR_i ≥ 2`; `cos(g1,g2) ≥ 0.3`; sign agreement ≥ 0.6; Spearman ≥ 0.3; `||g|| / bootstrap_SE(g) ≥ 2`; phenotype-change fraction ≥ 0.10.

### Q4 — Credit path

Does ACT consequence causally reach future action preference via **state-only** interventions on checkpoint clones?

Pass: every link survives and later actuator logits **and** frozen-RNG sampled behavior move in the predicted direction.  
A nonzero ΔW is not a pass. Do not edit the neural mechanism. “Responsible actuator” means the credited handle’s projection/logit under distributed motor vectors.

## Q1 optimizer (frozen)

- Type: Adam (`β1=0.9`, `β2=0.999`, `ε=1e-8`)
- Matrices: `W_in`, `W_rec` (masked by frozen `M`), `W_op`, `b_op`, `W_act_query`
- Init: packed v27 `make_cortex()` birth + bind; 3 restarts with `σ_init=0.01`
- Surrogate (stop-grad `ρ` after frozen sensory path on teacher-pair probe):

```text
L = −log p(ACT | ρ) − log p(h* | ACT, ρ)
    + λ_hold · p(HOLD | ρ)
    + λ_wrong · Σ_{h ≠ h*} p(h | ACT, ρ)
```

- `λ_hold=0.5`, `λ_wrong=0.5`, temperature 1
- Steps: 2000 max/restart; LR 1e-2 → 1e-4 cosine; clip `|W|≤2.0`; `W_rec *= M`
- Early stop: `probe_beneficial ≥ 0.80` for 20 consecutive evals (every 50 steps)
- Max compute: 4 FIT × 3 restarts × 2000 steps
- **Pass gate is behavioral `probe_beneficial`, never the surrogate**

## Q2 learning gates (frozen)

- One genotype; optimize only on `Q2.DIAG.FIT`; same genome must pass `Q2.DIAG.CHECK`
- Ordinary credit only; no Q1 weight inheritance; wired Arm D scalars only
- Unwired genes (`dyn.eligibility_decay`, `neuromod.gain.*`) are Q3 dead dimensions
- ≥ 4 CHECK worlds; sibling births; CI method `n_boot=9999`, seed `20260817`

## Q3 sample (frozen; also in runner.lock)

`P=32`, `σ=0.05`, `R=8` replicates, two independent batches, seed `20260817`.

## Q4 interventions (state-only)

- zero / swap `rho_elig`; previous / wrong-tick eligibility
- flip or zero world physics (host)
- snapshot/restore selected plastic tensors around one observe
- restore identical RNG state after clone
- Measure pre-sampling logits/probabilities and frozen-RNG sampled behavior

Links (beneficial and harmful): ACT→body; body→adv; adv→correct eligibility; credit→credited handle logit; later probe logits and behavior.

Pre-registered hypothesis (not a bug fix): equal-evidence HOLD when `len(ordered)≥2` before body change may suppress ACT.

## Diagnostic world domains

- `TM024.WALLMAP.Q1.DIAG.FIT.`
- `TM024.WALLMAP.Q1.DIAG.TRANSFER.`
- `TM024.WALLMAP.Q2.DIAG.FIT.`
- `TM024.WALLMAP.Q2.DIAG.CHECK.`
- `TM024.WALLMAP.Q3.DIAG.`
- `TM024.WALLMAP.Q4.DIAG.`

Failed DEV triplet 0 may be reused only as already-revealed diagnostic material. Never QUAL/EVAL.

## Order of freezes

1. This contract + prereg (commit/push before answers)
2. Runner + tests; ABI/synthetic smoke only
3. `docs/lineage_wallmap.runner.lock` on clean `origin/main`
4. Score Q4, Q1, Q3, then Q2
5. Decision lock

Any behavior-affecting runner change after the runner lock requires a versioned runner lock and fresh diagnostic worlds.

## Decision table

- **Q4 breaks** → precedence over Q2. Repair general credit path in a new architecture candidate; re-run a newly committed reachability diagnostic. Historical locks stay historical.
- **Q1 fails** → representation review (state structure, compartments, operations); capacity lane only after that, as general populations, not an L0 function.
- **Q1 passes; Q4 intact; Q2 fails** → plasticity, maturation, or replay architecture.
- **Q2 passes; Q3 fails** → search algorithm, structured genome, batching, variance reduction.
- **Everything passes** → new isolated lineage commitment with justified compute.

Do not increase n merely because L0 failed.

## Refuse

Scorer softening; L0-specific circuitry; QUAL/EVAL reveal; panel reuse as held-out; claiming impossibility; Arm D vs Arm C superiority; `earned_next`; 0.0.005; editing neural mechanism inside Q4; treating DIAG.FIT as held-out; treating a favorable birth as Q2 pass; variance in the Q3 SNR denominator; optimizing Q1 on sampled probe counts without the frozen surrogate; moving `τ`/`δ`.
