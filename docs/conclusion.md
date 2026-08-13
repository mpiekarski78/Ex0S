# Conclusion: three-memory v0

**Date:** 14 August 2026  
**Classification:** **Store-works**  
**Code:** [mpiekarski78/three-memory](https://github.com/mpiekarski78/three-memory)  
**BDH baseline:** [mpiekarski78/bdh](https://github.com/mpiekarski78/bdh) Category B (trace-only)

This is the deliverable for the first three-memory question. Evidence, not a product.

---

## Verdict

Frozen innate drives + write/retrieve rules **can** fill an inspectable world-knowledge store from experience. After ρ is reset, the fact still steers behavior **if and only if** it lives in S. With S disabled, the same experience behaves like BDH Category B: useful in-session, gone after reset.

Do not put a life of knowledge in ρ. Put it in S.

---

## Question

> Can frozen innate drives + learning rules fill an **inspectable** world-knowledge store from experience, such that facts **survive reset of the working trace** — while the trace alone does not?

| ID | Meaning | v0 |
|----|---------|----|
| Fail | Store junk / reset still kills the fact when S is on | no |
| Trace-only | ρ moves the next step; reset wipes it | **yes when S off** |
| Store-works | After experience, reset ρ, fact remains via S and is inspectable | **yes when S on** |
| Confound | Slow weights absorbed the fact | no (SHA256 unchanged) |

---

## Setup

| Item | Value |
|------|-------|
| World | key/door: `red door opens only with key` |
| Cortex | frozen random encoder + action head (seed 1337) |
| ρ | EMA embed + last successful action (session only) |
| S | JSON fact records; retrieve biases logits |
| Drives | novelty vs ρ; integrity-cost on failure |
| Controls | A vs B, disable-S, reset S, twin ρ, weight hash, ρ restore |

## Headline results (seed 12345)

| Condition | Probe `use_key` on red door with key |
|-----------|--------------------------------------|
| A after experience, before ρ reset | correct |
| A after ρ reset, S kept | **correct** |
| B (foil life) after ρ reset | incorrect (`open`) |
| disable-S before ρ reset | correct (session residue) |
| disable-S after ρ reset | **incorrect** |
| reset S then probe | incorrect |
| Weights unchanged | true |
| Twin ρ L2 | 0 |
| Fact in `store_A.json` | `"red door opens only with key"` |

---

## Implication

| Role | Where |
|------|-------|
| Species prior (sensors/dynamics) | Frozen cortex |
| “I just did this and it worked” | ρ, discard on reset |
| Beliefs / world facts / inspectable history | **S** |

Public BDH answered: ρ alone is Category B. This sibling answers: add an explicit store and the Category B ceiling lifts for **inspectable facts**, without pretending ρ is long-term memory.

Winning v0 does not reopen Category D on BDH ρ.

---

## Limitations (honest)

- This is a **tiny designed world**, not an emergent LLM. Store→action uses **tags** (`door=red` → prefer `use_key`), not natural-language understanding of `what`.
- The `what` string is a **fixed template** written by the learning rule when events fire; the environment no longer injects a labeled lesson string.
- Fallback forced curriculum (OPEN → PICK_KEY → USE_KEY) runs only if free policy never succeeds; reported in metrics as `n_forced_steps`.
- Foil B still forces `OPEN` on the blue door so A/B experience differs by design.
- Winning Store-works does **not** reopen Category D on BDH ρ.

## Reproduce

```bash
python -m experiments.run_v0
```

Numbers: `runs/` (gitignored). Comparison: [`comparison_bdh.md`](comparison_bdh.md). Protocol: [`protocol.md`](protocol.md).

## Gate

v0 is **Store-works** → v1 (tiny language, same three boxes) is **allowed**. See [`v1_plan.md`](v1_plan.md). Not started until explicitly run.
