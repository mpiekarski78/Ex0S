# Three-memory experiment protocol (v0)

## Question

> Can frozen innate drives + learning rules fill an **inspectable** world-knowledge store from experience, such that facts **survive reset of the working trace** — while the trace alone does not?

## Predeclared categories

| ID | Meaning |
|----|---------|
| Fail | Store writes junk, or behavior still dies when the trace is reset |
| Trace-only | Same as BDH Category B: ρ moves the next step; reset wipes it |
| Store-works | After experience, reset ρ, facts remain via the store and can be inspected |
| Confound | Slow weights silently absorbed the facts (illegal in v0) |

## Three pieces

1. **Frozen cortex** — species prior (sensors/dynamics). SHA256 must not change.
2. **Working trace ρ** — session EMA over embeddings. Reset is a first-class test.
3. **World store S** — explicit JSON records `{what, when, drive_scores, tags}`.

Innate drives (frozen thresholds): novelty / prediction-error vs ρ, integrity-cost on failure.
Write rule: write a lesson to S when novelty or integrity crosses threshold.
Retrieve rule: matching tags bias action logits (knowledge is not stored in ρ).

## v0 world

Fact: `red door opens only with key`.

- **A** experiences the contingency (OPEN fails, USE_KEY succeeds) with S on.
- **B** experiences a blue-door foil (no red-door fact).
- **disable-S** same experience as A, writes blocked.
- Probe: `probe_red_with_key` → correct action is `use_key`.

## Pass criteria (Store-works)

- A correct after ρ reset, S kept, and the fact is in `store_A.json`
- B incorrect on the same probe after ρ reset
- disable-S incorrect after ρ reset (BDH-like Category B)
- Resetting S removes the effect
- Weight hash unchanged
- Twin identical experience → ρ L2 ≈ 0

## Comparison to BDH

See [`comparison_bdh.md`](comparison_bdh.md). BDH is the trace-only baseline ([mpiekarski78/bdh](https://github.com/mpiekarski78/bdh)).
