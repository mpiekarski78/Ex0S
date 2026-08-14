# TM.0.7.0 results: retry-novel / shared return

**Date:** 14 August 2026  
**Classification:** **A Fail** · **B Store-works**  
**Run:** `runs/2026-08-14_225739_tm070`

Recipe jump: **do not lock find on the first max-novel page.** Same find-novel + two-rare clutter as TM.0.6.9. Here-only no longer freezes collect while an unowned page still adds a rare token vs the **whole** unread library, and a new novel page may be stamped even if this station already has a note. Novelty is not recomputed on the leftover pile (that restored a W walk). Train S is **4 files**: `c08`/`xenon`, `c09`/`neon`, `c10`/`krypton`, `p99`/`push`. C life adds `p98`/`adjust`. Probe **PRESS** / foil C **HOLD**. Wipe-between loses A. Cortex frozen (`a485b26b…`). `n_forced=0`. `n_train` was not raised. `MAX_TRAIN_S_FILES` stays 4. Retry-novel default **off**.

Skipped on purpose: dropping `has_code`, removing `domain=`, math, ranking `p98` by name, restoring unique-rare, leftover-pile rarity, raising `MAX_TRAIN_S_FILES`.

## Question

Once several unread pages share the max novel-count, can the store keep looking through that tie — or does here-only still freeze on the first tied page?

## Predeclared

| ID | A | B |
|----|---|----|
| Confound | Cortex moves; unique two-rare pair restored; leftover-pile rarity; `push` in the agent; drop `has_code` / `domain=`; ranker for `p98`; argon as clutter hapax; n_train raised | same |
| Fail | Untrained PRESS; train S not PRESS; train S still >4 files; C life loses A or misses TUNE; wipe-between still PRESS; bind=`xenon`/`neon`/`krypton`/`radon`/`lithium`/`cesium`/`nickel`/`cobalt`/`quartz`/`argon`; nonce PRESS | Untrained PRESS; A miss; C miss; n>4; nonce PRESS |
| Store-works | Two-rare-clutter English W; train S small; A PRESS / C HOLD from `push`; C life A PRESS / C TUNE from `adjust`; wipe-between A HOLD; nonce HOLD | Same without splitting the return |

## Headline

| Check | A split | B shared return |
|-------|---------|-----------------|
| Classification | **Fail** | **Store-works** |
| Untrained A / foil C | `hold` / `hold` | `hold` / `hold` |
| After train, S: A / foil C | **`press` / `hold`** | **`press` / `hold`** |
| C life on that S: A / C | `press` / `tune` | **`press` / `tune`** |
| Wipe-between A | **`hold`** | **`hold`** |
| Train S n files | **4** | **4** |
| n_revised train | 0 | 0 |
| Train S binds | `xenon` + `neon` + `krypton` + **`push`** | `xenon` + `neon` + `krypton` + `push` |
| C life binds | those + **`adjust`** | those + `adjust` |
| Train last 50 | 0.96 | 0.96 |
| Cortex | unchanged | unchanged |

A train notes are the whole novel tie on `w_a`. C added `p98` `bind=adjust` `did=tune`. Closed clutter-only HOLD. Nonce HOLD. Common leftover pages (`c00`–`c07`) stayed out of train S.

## Compare

**A** is the jump: TM.0.6.9 locked on the first tied page (`c09` / `neon` only). Retry-novel keeps looking until every full-W novel page is owned. S now holds `push` and later `adjust`, and also the three clutter hapax. The English two-station bar fails on those binds. Do not restore leftover-pile rarity. Do not rank `p98`.

**B** shared return **Store-works** on the motor bar (n=4, PRESS/TUNE, last-50 0.96). Not the jump. Not retuned.

## Audit (not retuned)

- TM.0.5.0–0.7.0 unit tests still pass.
- Default door agent: retry-novel / find-novel / in-hand new-here / revise / here-only / block-here / stamp-new-here / one-bind **off**.
- TM.0.6.0–0.6.9 `make()` leave retry-novel off. TM.0.6.9 A remains Fail without it; its train S stays one file.
- Search still computes `has_code`. `domain="dial"` still set. No English lexicon in the agent.

## Honest limits

- Stream-first is still frozen grammar, not English syntax.
- Tiny closed corpus, not Wikipedia.
- The store now has more language (`push` + `adjust` + clutter hapax). It still cannot prefer `push` over `xenon`.
- Novelty for retry is vs the whole library. Recomputing rares on the leftover pile walks common pages; that path is off.
- Split credit still does not teach which tied page is English.
- Shared return passing here does not rewrite this Fail or earlier Fail slices.
- Still five acts, innate names, `domain=`. Math is a later life.

## Reproduce

```bash
python tests/test_tm070.py
python tests/test_tm069.py
python tests/test_tm068.py
python tests/test_tm067.py
python tests/test_tm066.py
python tests/test_tm065.py
python tests/test_tm064.py
python tests/test_tm063.py
python tests/test_tm062.py
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
python -m experiments.run_tm070
```
