# TM.0.5.5 results: accumulate S / shared return

**Date:** 14 August 2026  
**Classification:** **A Store-works** · **B Fail**  
**Run:** `runs/2026-08-14_131646_tm055`

Recipe jump: **accumulate S** (eval path). One unread library with both useful pages plus distinct Open W clutter. Life A then life C, **same S**, no wipe. After both, ρ reset, W gone: probe A **PRESS** (first fact kept); probe C **TUNE** (second fact added). Wipe-between: A **HOLD** / C **TUNE**. After A only: A PRESS / C HOLD. Copy-only still PRESS on C. Train still wipes each episode. Cortex frozen (`a485b26b…`). `n_forced=0`.

Skipped on purpose: dropping `has_code`, solving B, removing `domain=`, never-wipe train, English.

## Question

Can two lives share one store, or does the second life require wiping the first?

## Predeclared

| ID | A | B |
|----|---|----|
| Confound | Cortex moves; eval W missing both pages; English; train also never wipes; drop `has_code` / `domain=` | same |
| Fail | S wiped between eval lives; after both, A lost or C miss; wipe-between still PRESS on A | Untrained PRESS; two-life miss; split restored |
| Store-works | After A: A PRESS / C HOLD; after both: A PRESS / C TUNE; wipe-between A HOLD; copy-only PRESS on C | Same without split |

## Headline

| Check | A split | B shared return |
|-------|---------|-----------------|
| Classification | **Store-works** | **Fail** |
| Untrained A / foil C | `hold` / `hold` | `hold` / `hold` |
| After A life: A / foil C | **`press` / `hold`** | `hold` / `hold` |
| After both lives: A / C | **`press` / `tune`** | `hold` / `hold` |
| Wipe-between: A / C | **`hold` / `tune`** | `hold` / `hold` |
| Copy-only foil C | **`press`** (cheat) | `hold` (never stamped) |
| Train last 50 | 0.84 | **0.00** |
| Cortex | unchanged | unchanged |

## Compare

**A** is the jump: yesterday’s file is still there. Same S, two facts, here-match picks the station. Wipe-between loses A, so keeping S is load-bearing.

**B** shared return **Fail** (last-50 0). Same credit hole.

Genome grammar added for two facts (closed body vocabulary, not English): skip owned W pages only when S already names a **different** station; do not stamp a note that names another station; retrieve the S file that names **here**. Door agents unchanged (`use_here_match` off).

## Audit (not retuned)

- TM.0.5.0–0.5.4 unit tests still pass.
- Search still computes `has_code`. `domain="dial"` still set. Train still wipes every episode. No English lexicon (`push` absent). W has no motor or station names.

## Honest limits

- Train still wipes. This slice is eval accumulate, not a never-wipe organism.
- Here-filter is frozen grammar on innate station names. Not a learned ranker, not Wikipedia.
- Still five discrete acts, rarity, split credit. Shared return still fails.

## Reproduce

```bash
python tests/test_tm055.py
python tests/test_tm054.py
python tests/test_tm053.py
python tests/test_tm052.py
python tests/test_tm051.py
python tests/test_tm050.py
python -m experiments.run_tm055
```
