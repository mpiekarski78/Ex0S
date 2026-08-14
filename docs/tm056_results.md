# TM.0.5.6 results: never-wipe train / shared return

**Date:** 14 August 2026  
**Classification:** **A Store-works** · **B Fail**  
**Run:** `runs/2026-08-14_133451_tm056`

Recipe jump: **never-wipe train.** ρ still resets each episode. S is not deleted. After 500 A lives, that dirty store (no fresh A life) probes **PRESS** / foil C **HOLD**. C life on the same dirty S: A **PRESS** kept, C **TUNE** added. Wipe-between loses A. Cortex frozen (`a485b26b…`). `n_forced=0`. Train S had 10 files.

Skipped on purpose: dropping `has_code`, solving B, removing `domain=`, English. `use_commit_rare_only` is on for this slice only (default off).

## Question

Can the fact written during training survive 500 more episodes in the same store, or does learning still need the experimenter to wipe S?

## Predeclared

| ID | A | B |
|----|---|----|
| Confound | Cortex moves; train still wipes; English; drop `has_code` / `domain=`; rare-commit on by default | same |
| Fail | Dirty train S not PRESS; C life loses A or misses TUNE; wipe-between still PRESS on A | Untrained PRESS; dirty S miss; split restored |
| Store-works | After train A PRESS / C HOLD; C on that S A PRESS / C TUNE; wipe-between A HOLD | Same without split |

## Headline

| Check | A split | B shared return |
|-------|---------|-----------------|
| Classification | **Store-works** | **Fail** |
| Untrained A / foil C | `hold` / `hold` | `hold` / `hold` |
| After train, dirty S: A / foil C | **`press` / `hold`** | `hold` / `hold` |
| C life on dirty S: A / C | **`press` / `tune`** | `hold` / `hold` |
| Wipe-between: A / C | **`hold` / `hold`** | `hold` / `hold` |
| Train S n files | 10 | 12 |
| Train last 50 | 0.90 | **0.00** |
| Cortex | unchanged | unchanged |

## Compare

**A** is the jump: the store is the organism. Training does not throw it away. A later C life writes a second fact without wiping the first.

**B** shared return **Fail** (last-50 0). Same credit hole.

`use_commit_rare_only` (this slice): once S already names **here**, do not commit non-rare W pages. Default **off** so TM.0.5.5 B stays Fail.

## Audit (not retuned)

- TM.0.5.0–0.5.5 unit tests still pass.
- TM.0.5.5 experiment rerun **Store-works** / **Fail** (`runs/2026-08-14_133616_tm055`).
- Search still computes `has_code`. `domain="dial"` still set. No English lexicon (`push` absent).

## Honest limits

- Train S is dirty (~10 files), not a tidy notebook. Here-filter finds `cha` among them.
- Fresh empty-S copy-only no longer PRESS: never-wipe policy is not required to re-find from a blank store. Here-match load-bearing is A PRESS / C HOLD on the dirty store.
- Still five acts, rarity, split credit. Shared return still fails. Not Wikipedia.

## Reproduce

```bash
python tests/test_tm056.py
python tests/test_tm055.py
python tests/test_tm054.py
python tests/test_tm053.py
python tests/test_tm052.py
python tests/test_tm051.py
python tests/test_tm050.py
python -m experiments.run_tm056
```
