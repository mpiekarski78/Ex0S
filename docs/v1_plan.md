# v1 plan (gated): tiny language, same three-memory split

**Gate:** v0 classified **Store-works** → this phase was allowed.  
**Status:** **done** — [`v1_results.md`](v1_results.md). Classification **Store-works**.

## Goal

Same three boxes as v0 (frozen cortex / prior, session ρ, inspectable S), with observations as bytes/tokens.

Question: can experience write **language facts** into S such that, after ρ reset, completions still reflect those facts — while disable-S recovers BDH Category B on the same probes?

**Answer:** yes, on this tiny LSTM + NOTE retrieve. disable-S after ρ reset matches the empty prior (BDH-like B). S-on keeps P(`v`)≈0.99 after reset.

## What shipped

1. Pretrain tiny byte LM on stripped Shakespeare + NOTE-copy (`experiments.train_prior`).
2. Freeze weights. Experience writes 5-grams to S.
3. Retrieve longest suffix match as `NOTE:` context.
4. Controls: A/B, disable-S, reset ρ, reset S, weight hash, twins, 1-byte filler.
5. Comparison uses **published** BDH numbers (read-only). No `bdh.py` import.
