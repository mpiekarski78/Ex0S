# TM.0.5.3 results: use-the-fact / shared return

**Date:** 14 August 2026  
**Classification:** **A Store-works** · **B Fail**  
**Run:** `runs/2026-08-14_125314_tm053`

Recipe jump: **use the fact, not pick-a-motor.** Same unnamed W as TM.0.5.2. On a real success the body stamps its act **and** its innate station name (`cha` / `chc`) onto the rare note. After A life, **same S**, ρ reset, W gone: probe A **PRESS**; probe C **HOLD**. After a separate C life: C **TUNE**, A **HOLD**. Copy-only (here-match off) still PRESS on C. Cortex frozen (`a485b26b…`). `n_forced=0`.

Skipped on purpose: dropping `has_code`, solving B, removing `domain=`, accumulating S, Open W, English.

## Question

After ρ reset, is the committed file a fact about **this station**, or a global motor to fire anywhere?

## Predeclared

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; station/motor name in W; Open W; English; accumulate S; drop `has_code` / `domain=` | same |
| Fail | Untrained PRESS; A's `press` fires on C; copy-only already HOLDs on C | Untrained PRESS; A miss; split restored |
| Store-works | Untrained HOLD; A life → A PRESS / C HOLD; C life → C TUNE / A HOLD; copy-only PRESS on C | Same without split |

## Headline

| Check | A split | B shared return |
|-------|---------|-----------------|
| Classification | **Store-works** | **Fail** |
| Untrained A / foil C | `hold` / `hold` | `hold` / `hold` |
| After A life: A / foil C | **`press` / `hold`** | `hold` / `hold` |
| After C life: C / foil A | **`tune` / `hold`** | `hold` / `hold` |
| Copy-only foil C | **`press`** (cheat) | `hold` (never stamped) |
| S after A life | `p99` `w7=press` `w8=cha` | `c00` clutter |
| Train last 50 | 0.34 | **0.00** |
| Cortex | unchanged | unchanged |

## Compare

**A** is the jump: the stamp is not a motor you always fire. Same S, wrong station → HOLD. Copy-only without here-match still PRESS on C, so the match is load-bearing.

**B** shared return **Fail** (last-50 0). Same credit hole.

## Audit (not retuned)

- TM.0.5.0 still **Store-works** / **Fail** (`runs/2026-08-14_125353_tm050`).
- TM.0.5.1 still **Store-works** / **Fail** (`runs/2026-08-14_125358_tm051`).
- TM.0.5.2 still **Store-works** / **Fail** (`runs/2026-08-14_125424_tm052`).
- Search still computes `has_code`. `domain="dial"` still set. S still wiped every train episode. W still 12 short scraps. No English lexicon (`push` absent).

## Honest limits

- Genome knows innate **station** names (`cha` / `chb` / `chc`) the same way it knows act names. Not English.
- Here-match is frozen grammar: copy the selected act only if the file names *here*.
- Still five discrete acts, rarity, split credit. Not Wikipedia. Shared return still fails.

## Reproduce

```bash
python tests/test_tm053.py
python tests/test_tm052.py
python tests/test_tm051.py
python tests/test_tm050.py
python -m experiments.run_tm053
```
