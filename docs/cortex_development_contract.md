# TM.0.23.CORTEX.DEVELOP — developmental phenotype contract

Product under test: **Ex0S 0.0.004**. This contract freezes D0–D12 scoring rules for continuous lives on frozen `make_cortex`. No neural mechanism change. Eligibility for a later 0.0.005 stamp is reported; this pass does not stamp.

## Stance

- One organism per birth seed; no capability flags between stages.
- Teaching schedules come only from fixtures and the frozen world transition kernel (`physics`).
- `scorer_only` never enters `observe`, `body_state`, or action credit.
- Latent world state may drive frozen physics → observable consequence → cortex.
- Runner must not inspect failures and dynamically add helpful demonstrations.
- `earned_next=false`; `ex0s=null`; `product=0.0.004`.

## Required vs diagnostic

| Work | Role |
|------|------|
| D0–D12 paired lives | **Required** for `development_gate_clear` (≥13/16 pairs clear all required D stages) |
| D10 maturation | **Additional** ≥14/16 adult outperforms paired child for `eligible_for_000005` |
| D12 cortex/S separation forks | **Required**; launching capacity alone does **not** clear D12 |
| Capacity lanes | **Diagnostic**; not part of initial eligibility |
| Neural parity wall | **Diagnostic**; need not pass; social-wall failure cannot negate a cleared neural gate |

## D0 — Birth absence (population-level)

A single correct random action at birth is neither a pass nor an automatic failure.

### Frozen chance model (before reveal)

- Probe count `n_probes = 64`
- Each probe is a counterbalanced binary preference trial between a **future-curriculum target** token and a **foil** token (order randomized per probe via `rng_permute` stream derived from birth seed).
- Null chance `p0 = 0.5`
- Statistic: one-sided binomial test of target preference rate `k/n` against `H0: p = p0`
- Confidence: reject H0 (above-chance knowledge) if `P(X ≥ k | Binomial(n,p0)) < α` with **`α = 0.01`**
- **D0 passes** iff all hold:
  1. Target preference is **not** significantly above chance (fail to reject H0 at α).
  2. No fixture-specific answer dependence: swapping which token is labeled “target” in the scorer does not invert a significant preference already present at birth (symmetric absence).
  3. S is empty; no preloaded emit sequences matching curriculum targets.
- HOLD or undirected exploration is allowed.
- Always-HOLD fails **D1 onward**, not D0 by itself.

## D1 — Sensorimotor contingency

After teaching ACT→physics consequences, organism’s ACT operand distribution shifts toward beneficial operands; counterfactual operand swap changes predicted/observed consequence path. No evaluator correctness scalar.

## D2 — Association and revision

Opaque symbol–consequence association forms; contradictory experience increases uncertainty/HOLD; further experience revises; ρ reset preserves matured association behavior.

## D3 — Relation under distractors

Repeated multi-symbol relation; ordinary distractors not durable; equal evidence → HOLD; renamed twin follows twin experience.

## D4 — External memory use

One-shot fact via WRITE/S; recall after interference + ρ reset; strip relevant S → fact gone; donor S changes recalled fact.

## D5 — Symbol grounding

Ground via observable action/consequence; unknown/ambiguous → HOLD; meaning swap follows experience; no answer-menu-only path.

## D6 — Ordered construction

Emit 1–2 symbol sequences; ordered role effects; STOP from experience; ambiguous next/STOP → atomic HOLD.

## D7 — Variable-length sequence

Lengths 1→2→4→8; length never supplied by host; no partial-credit prefixes; cap without learned STOP → HOLD.

## D8 — Novel composition

Components/orders experienced; exact target withheld from teaching; construct target; twin follows own experience.

## D9 — Retention

Continue new domains; earlier skills remain; returning to old domain does not reload a specialized mechanism.

## D10 — Maturation

Compare birth → early child → intermediate → mature checkpoints. Adult outperforms own child; improvement not from more answer candidates; ρ reset preserves slow maturation. Population gate: ≥14/16 pairs.

## D11 — Renamed developmental twin

Independent symbol vectors, renamed tokens, permuted order, altered schedule, equivalent structure; behavior follows twin experience.

## D12 — Cortex/S separation (required forks)

Must pass:

1. mature cortex + stripped arbitrary-fact rows → facts lost, general acquisition skill retained
2. birth cortex + adult S → information present but immediate use worse than mature cortex
3. donor S → facts follow donor
4. donor mature cortex + compatible fresh S → learning behavior follows cortical competence
5. reset cortical weights while retaining S → developmental competence decreases
6. reset ρ only → mature competence remains

Capacity lanes and neural parity wall are **launched after** D12 forks; they do not substitute for D12.

## Earn / eligibility fields

```text
development_gate_clear  ≥13/16 pairs clear D0–D12 (incl. D12 forks)
eligible_for_000005     development_gate_clear AND maturation ≥14/16
earned_next             false
ex0s                    null
product                 0.0.004
```

## Stats

- Exactly 16 main/twin pairs (optional extra 16 never required)
- No seed replacement
- Report per-stage success distributions
