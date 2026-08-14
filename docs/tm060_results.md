# TM.0.6.0 results: first English life / shared return

**Date:** 14 August 2026  
**Classification:** **A Store-works** · **B Store-works**  
**Run:** `runs/2026-08-14_150422_tm060`

Recipe jump: **first English life, tiny corpus.** Unread pages never say `press`/`tune`. Genome has no synonym table. On success the body writes `did=` (bookkeeping, not a copy token) and keeps rare page words; after ρ reset, W gone, A **PRESS** / C **TUNE** from those words via inspectable aliases in S. Untrained **HOLD**. Bind-off **HOLD**. Cortex frozen (`a485b26b…`). `n_forced=0`. `n_train` was not raised. New flags default **off**.

Skipped on purpose: dropping `has_code`, removing `domain=`, math, solving B (B happened to pass; not retuned).

## Question

Can a free life over English pages bind a corpus word to a motor without putting that synonym in DNA?

## Predeclared

| ID | A | B |
|----|---|----|
| Confound | Cortex moves; W names an innate motor; nonce scrap instead of English; `push` in the agent; drop `has_code` / `domain=`; innate name still a copy token in S | same |
| Fail | Untrained PRESS; after A life not PRESS / foil C not HOLD; after C life not TUNE; bind-off still PRESS; annotate-off still PRESS | Untrained PRESS; A miss; C miss; split restored |
| Store-works | Untrained HOLD; after a free life A PRESS / C HOLD from `push` bound in S (`did=press`, not `w*=press`); C life TUNE from `adjust`; bind-off HOLD | Same without split |

## Headline

| Check | A split | B shared return |
|-------|---------|-----------------|
| Classification | **Store-works** | **Store-works** |
| Untrained A / foil C | `hold` / `hold` | `hold` / `hold` |
| After A life: A / foil C | **`press` / `hold`** | **`press` / `hold`** |
| After C life: C / foil A | **`tune` / `hold`** | **`tune` / `hold`** |
| Bind-off A | **`hold`** | **`hold`** |
| Copy-only foil C | `press` | `press` |
| Train last 50 | 0.88 | 0.46 |
| Cortex | unchanged | unchanged |

Inspectable A note: `did=press`, `w*=turned|up|push|cha`. No `press` as a copy token.

## Compare

**A** is the jump: TM.0.5.2 stamped the innate name `press` onto S and copied it. Here the page says `push`; S copies `push`; `did=press` is bookkeeping the probe does not read. Bind-off (stamp `did=`, no alias lookup) stays HOLD, so the bind is load-bearing.

**B** shared return **Store-works** on this slice (last-50 0.46). Not the jump. Not retuned.

## Audit (not retuned)

- TM.0.5.0–0.5.9 unit tests still pass.
- Default door agent: here-match / annotate / rare-commit / here-only / revise / alias-bind / did-stamp **off**.
- TM.0.5.4 / TM.0.5.9 `make()` leave alias-bind and did-stamp off.
- Search still computes `has_code`. `domain="dial"` still set. No English lexicon in the agent (`push` / `adjust` absent).

## Honest limits

- Tiny closed corpus (11 clutter + 2 pages), not Wikipedia.
- Bind is frozen grammar: rare tokens on a successful note name the act just done. `turned` and `up` are aliased too. That is not English NLP.
- Search is still binary `{has_code, has_rare}`.
- Shared return passing here does not rewrite earlier Fail slices and does not drop split credit.
- Still five acts, innate names, `domain=`. Math is a later life.

## Reproduce

```bash
python tests/test_tm060.py
python tests/test_tm059.py
python tests/test_tm058.py
python tests/test_tm057.py
python tests/test_tm056.py
python tests/test_tm055.py
python tests/test_tm054.py
python tests/test_tm053.py
python tests/test_tm052.py
python tests/test_tm051.py
python tests/test_tm050.py
python -m experiments.run_tm060
```
