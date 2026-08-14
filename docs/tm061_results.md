# TM.0.6.1 results: one bind / shared return

**Date:** 14 August 2026  
**Classification:** **A Store-works** · **B Store-works**  
**Run:** `runs/2026-08-14_153813_tm061`

Recipe jump: **one bind per successful note.** Same tiny English life as TM.0.6.0, but the useful page has two rare words (`push` and `argon`). Genome aliases the first rare token in **stream order**, not every hapax. After ρ reset, W gone: A **PRESS** / C **TUNE**. Argon stays on the note and does **not** fire. Bind-all on argon-only S **PRESS** (load-bearing). Untrained **HOLD**. Bind-off **HOLD**. Cortex frozen (`a485b26b…`). `n_forced=0`. `n_train` was not raised. `use_one_bind` default **off**.

Skipped on purpose: dropping `has_code`, removing `domain=`, math, solving B (B happened to pass; not retuned).

## Question

Can a page be many rare words, with only one bound name for the act — without a synonym table in DNA?

## Predeclared

| ID | A | B |
|----|---|----|
| Confound | Cortex moves; unique-rare needle; W names an innate motor; `push` in the agent; drop `has_code` / `domain=`; alphabet bind of `argon` | same |
| Fail | Untrained PRESS; after A life not PRESS; bind=`argon`; nonce-only S PRESS; bind-all on nonce HOLD | Untrained PRESS; A miss; C miss; nonce PRESS |
| Store-works | Untrained HOLD; two rares on the page; bind=`push` not `argon`; A PRESS / C HOLD; nonce HOLD; bind-all nonce PRESS; C life TUNE from `adjust` | Same without split |

## Headline

| Check | A split | B shared return |
|-------|---------|-----------------|
| Classification | **Store-works** | **Store-works** |
| Untrained A / foil C | `hold` / `hold` | `hold` / `hold` |
| After A life: A / foil C | **`press` / `hold`** | **`press` / `hold`** |
| After C life: C / foil A | **`tune` / `hold`** | **`tune` / `hold`** |
| Bind-off A | **`hold`** | **`hold`** |
| Nonce-only A | **`hold`** | **`hold`** |
| Bind-all nonce A | **`press`** | **`press`** |
| Train last 50 | 0.88 | 0.90 |
| Cortex | unchanged | unchanged |

Inspectable A note: `bind=push`, `did=press`, `w*=argon|push|cha`. `argon` sorts before `push`; stream order bound `push`.

## Compare

**A** is the jump: TM.0.6.0 aliased every hapax (`turned|up|push`). Here `argon` is on the same note and does not mean PRESS. Bind-all on an argon-only copy of S still PRESS, so one-bind is load-bearing.

**B** shared return **Store-works** on this slice (last-50 0.90). Not the jump. Not retuned.

## Audit (not retuned)

- TM.0.5.0–0.6.0 unit tests still pass.
- Default door agent: alias-bind / did-stamp / one-bind **off**.
- TM.0.6.0 `make()` leaves one-bind off.
- Search still computes `has_code`. `domain="dial"` still set. No English lexicon in the agent (`push` / `argon` / `adjust` / `alpha` absent).

## Honest limits

- Stream-first is frozen grammar (symbol streams have order), not English syntax. The page puts the synonym before the nonce.
- Tiny closed corpus, not Wikipedia.
- Search is still `{has_code, has_rare}`.
- Shared return passing here does not rewrite earlier Fail slices.
- Still five acts, innate names, `domain=`. Math is a later life.

## Reproduce

```bash
python tests/test_tm061.py
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
python -m experiments.run_tm061
```
