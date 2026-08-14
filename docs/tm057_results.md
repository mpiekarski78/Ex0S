# TM.0.5.7 results: find without unique rare / shared return

**Date:** 14 August 2026  
**Classification:** **A Store-works** · **B Fail**  
**Run:** `runs/2026-08-14_134833_tm057`

Recipe jump: **find without a unique rare token.** Open W has three hapax clutter pages (`xenon` / `argon` / `neon` on `c08`–`c10`) plus krypton/helium. `has_rare` is no longer a unique pointer at p99. After never-wipe train the dirty store (6 files, including several distinctive pages) probes **PRESS** / foil C **HOLD**. C life on that S: A **PRESS** kept, C **TUNE** added. Wipe-between loses A. Cortex frozen (`a485b26b…`). `n_forced=0`. Closed-lexicon clutter-only (no hapax) stays HOLD.

Skipped on purpose: dropping `has_code`, solving B, removing `domain=`, English.

## Question

Can search still find a stampable page when several unread documents are distinctive, or does the recipe need one unique rare needle?

## Predeclared

| ID | A | B |
|----|---|----|
| Confound | Cortex moves; useful page is still the only rare token; English; drop `has_code` / `domain=` | same |
| Fail | Dirty train S not PRESS; C life loses A or misses TUNE; wipe-between still PRESS on A; closed clutter-only PRESS | Untrained PRESS; dirty S miss; split restored |
| Store-works | Multi-rare W; after train A PRESS / C HOLD; C on that S A PRESS / C TUNE; wipe-between A HOLD | Same without split |

## Headline

| Check | A split | B shared return |
|-------|---------|-----------------|
| Classification | **Store-works** | **Fail** |
| Untrained A / foil C | `hold` / `hold` | `hold` / `hold` |
| After train, dirty S: A / foil C | **`press` / `hold`** | `press` / `hold` |
| C life on dirty S: A / C | **`press` / `tune`** | `press` / `hold` |
| Wipe-between: A / C | **`hold` / `hold`** | `hold` / `hold` |
| Rare clutter pages | 3 | 3 |
| Train S n files | 6 | 8 |
| Train last 50 | 0.92 | **0.86** |
| Cortex | unchanged | unchanged |

## Compare

**A** is the jump: uniqueness was a cheat. Several distinctive lab logs, same `{has_code, has_rare}` recipe, never-wipe store. S committed xenon, argon, neon, **and** krypton, then stamped `press`+`cha`. A later C life added helium and `tune`+`chc`.

**B** shared return **Fail**: C life on dirty S stayed HOLD (no `tune` stamp). Last-50 **0.86** — one return can now train the first fact because any distinctive page is a blank, but the second fact still misses. Not Store-works. Not retuned.

## Audit (not retuned)

- TM.0.5.0–0.5.6 unit tests still pass.
- No agent flag added. Default door agent still has here-match / annotate / rare-commit off.
- Search still computes `has_code`. `domain="dial"` still set. No English lexicon (`push` absent).

## Honest limits

- Search is still binary `{has_code, has_rare}`. Several hapaxes, not Wikipedia ranking.
- Distinctive pages are stampable blanks. The genome does not read krypton as a fact.
- Shared return last-50 is no longer 0, but C still misses TUNE. Split credit stays load-bearing for two facts.
- Still five acts, innate names, `domain=`. Short logs, not Wikipedia.

## Reproduce

```bash
python tests/test_tm057.py
python tests/test_tm056.py
python tests/test_tm055.py
python tests/test_tm054.py
python tests/test_tm053.py
python tests/test_tm052.py
python tests/test_tm051.py
python tests/test_tm050.py
python -m experiments.run_tm057
```
