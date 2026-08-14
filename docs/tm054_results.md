# TM.0.5.4 results: Open W / shared return

**Date:** 14 August 2026  
**Classification:** **A Store-works** · **B Fail**  
**Run:** `runs/2026-08-14_130209_tm054`

Recipe jump: **Open W.** Unread pages are document-shaped (heading plus a few paragraphs). Clutter is **11 distinct** bench logs from a closed lexicon, not 11 clones of one line. Useful fact still unnamed (no `press`/`tune`/`cha`). Same find/stamp/here-match as TM.0.5.3. After A life, **same S**, ρ reset, W gone: probe A **PRESS**; probe C **HOLD**. After a separate C life: C **TUNE**, A **HOLD**. Copy-only (here-match off) still PRESS on C. Cortex frozen (`a485b26b…`). `n_forced=0`.

Skipped on purpose: dropping `has_code`, solving B, removing `domain=`, accumulating S, English.

## Question

Does the same find/stamp/here-match recipe still work when unread W is a small stack of distinct documents, not cloned one-liners?

## Predeclared

| ID | A | B |
|----|---|----|
| Confound | Cortex moves; cloned one-line W; clutter pages are rare; station/motor name in W; English; accumulate S; drop `has_code` / `domain=` | same |
| Fail | W not Open W; untrained PRESS; A's `press` fires on C; copy-only already HOLDs on C | Untrained PRESS; A miss; split restored |
| Store-works | Distinct multi-paragraph W; untrained HOLD; A life → A PRESS / C HOLD; C life → C TUNE / A HOLD; copy-only PRESS on C | Same without split |

## Headline

| Check | A split | B shared return |
|-------|---------|-----------------|
| Classification | **Store-works** | **Fail** |
| Untrained A / foil C | `hold` / `hold` | `hold` / `hold` |
| After A life: A / foil C | **`press` / `hold`** | `hold` / `hold` |
| After C life: C / foil A | **`tune` / `hold`** | `hold` / `hold` |
| Copy-only foil C | **`press`** (cheat) | `hold` (never stamped) |
| S after A life | `p99` `w17=press` `w18=cha` | `c00` clutter |
| W clutter | 11 distinct, ≥2 paragraphs, not rare | same library |
| Train last 50 | 0.14 | **0.00** |
| Cortex | unchanged | unchanged |

## Compare

**A** is the jump: the recipe did not need a new head. Distinct documents, same rarity search, same stamp+here-match. Same S, wrong station → HOLD.

**B** shared return **Fail** (last-50 0). Same credit hole. B's A life committed clutter `c00`.

Train last-50 on A is 0.14 (was 0.34 on cloned scraps in TM.0.5.3). Eval probes still pass. Not retuned.

## Audit (not retuned)

- TM.0.5.0–0.5.3 unit tests still pass. Agent and dial world were not changed for this jump.
- Search still computes `has_code`. `domain="dial"` still set. S still wiped every train episode. No English lexicon (`push` absent). W has no motor or station names.

## Honest limits

- W is twelve short lab logs, not Wikipedia. Closed clutter lexicon keeps `has_rare` load-bearing.
- Genome still knows innate act and station names. Here-match is frozen grammar.
- Still five discrete acts, rarity, split credit. Shared return still fails.
- Harder W lowered last-50; we did not raise `n_train` or `lr` to rescue the plot.

## Reproduce

```bash
python tests/test_tm054.py
python tests/test_tm053.py
python tests/test_tm052.py
python tests/test_tm051.py
python tests/test_tm050.py
python -m experiments.run_tm054
```
