# TM.0.5.8 results: scale of Open W / shared return

**Date:** 14 August 2026  
**Classification:** **A Store-works** · **B Store-works**  
**Run:** `runs/2026-08-14_135627_tm058`

Recipe jump: **scale of Open W.** Clutter is 64 distinct multi-paragraph logs, not 11. Same multi-rare never-wipe recipe as TM.0.5.7 (`xenon`/`argon`/`neon` plus krypton/helium). After train the dirty store probes **PRESS** / foil C **HOLD**. C life on that S: A **PRESS** kept, C **TUNE** added. Wipe-between loses A. Cortex frozen (`a485b26b…`). `n_forced=0`. Closed-lexicon clutter-only (64 pages, no hapax) stays HOLD. `n_train` was not raised.

Skipped on purpose: dropping `has_code`, removing `domain=`, English. Shared return was not the jump; it is reported as measured.

## Question

Does the same find/stamp/never-wipe recipe still work when unread W is a pile of documents, or does it need a dozen-page toy library?

## Predeclared

| ID | A | B |
|----|---|----|
| Confound | Cortex moves; W still a dozen logs; useful page is the only rare token; English; drop `has_code` / `domain=` | same |
| Fail | Dirty train S not PRESS; C life loses A or misses TUNE; wipe-between still PRESS on A; closed clutter-only PRESS | Untrained PRESS; dirty S miss; split restored |
| Store-works | 64-page W; after train A PRESS / C HOLD; C on that S A PRESS / C TUNE; wipe-between A HOLD | Same without split |

## Headline

| Check | A split | B shared return |
|-------|---------|-----------------|
| Classification | **Store-works** | **Store-works** |
| Untrained A / foil C | `hold` / `hold` | `hold` / `hold` |
| After train, dirty S: A / foil C | **`press` / `hold`** | **`press` / `hold`** |
| C life on dirty S: A / C | **`press` / `tune`** | **`press` / `tune`** |
| Wipe-between: A / C | **`hold` / `tune`** | `hold` / `hold` |
| Distinct clutter | 64 | 64 |
| Rare clutter pages | 3 | 3 |
| Train S n files | 19 | 22 |
| Train last 50 | 0.90 | 0.94 |
| Cortex | unchanged | unchanged |

## Compare

**A** is the jump: a 64-page document pile, same `{has_code, has_rare}` recipe, never-wipe store. Untrained first-file is still common clutter. Trained search still lands on a distinctive page.

**B** shared return **Store-works** on this slice (last-50 0.94; C life stamped `tune`). That was not the jump and was not retuned. TM.0.5.7 B on the dozen-page library still **Fail**ed (C HOLD). Do not read this as “one return is solved.”

## Audit (not retuned)

- TM.0.5.0–0.5.7 unit tests still pass.
- No agent flag added. Default door agent still has here-match / annotate / rare-commit off.
- Search still computes `has_code`. `domain="dial"` still set. No English lexicon (`push` absent).
- `n_train` stayed 500.

## Honest limits

- Search is still binary `{has_code, has_rare}`. 64 short logs, not Wikipedia ranking.
- Distinctive pages are still stampable blanks. Train S is dirtier (19 files) than TM.0.5.7.
- Shared return working here does not rewrite TM.0.5.3–0.5.7. Split credit is still the crutch on those slices.
- Still five acts, innate names, `domain=`.

## Reproduce

```bash
python tests/test_tm058.py
python tests/test_tm057.py
python tests/test_tm056.py
python tests/test_tm055.py
python tests/test_tm054.py
python tests/test_tm053.py
python tests/test_tm052.py
python tests/test_tm051.py
python tests/test_tm050.py
python -m experiments.run_tm058
```
