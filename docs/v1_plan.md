# v1 plan (gated): tiny language, same three-memory split

**Gate:** v0 classified **Store-works** → this phase is allowed.  
**Not started** until someone runs it. Do not retune v0 to chase language plots.

## Goal

Same three boxes as v0 (frozen cortex / prior, session ρ, inspectable S), with observations as bytes/tokens.

Question: can experience write **language facts** into S such that, after ρ reset, completions still reflect those facts — while disable-S recovers BDH Category B on the same probes?

## Design (draft)

1. Pretrain a tiny byte LM on syntax/dynamics only (no probe facts in the corpus).
2. Freeze those weights (species prior).
3. Experience writes to S via the same novelty / integrity rules (salient mismatches, failed predictions).
4. Retrieve from S into the conditioning context (explicit string), never into ρ as the sole store.
5. Controls: A vs B, disable-S, reset ρ, reset S, weight hash.

## Comparable numbers (vs BDH)

Only in v1 — side by side with the existing BDH checkpoint **read-only**:

- `my lord` / `my love`, probe `my lo`
- empty prior, ΔP, JS
- 1-byte clean filler
- ρ reset

Until then, do **not** claim “better than BDH on Shakespeare.”

## Out of scope

- Modifying Pathway / mpiekarski78 `bdh.py`
- Treating ρ as Category D
- Shipping a chatbot
