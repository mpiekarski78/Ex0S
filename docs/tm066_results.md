# TM.0.6.6 results: correct dirty English S / shared return

**Date:** 14 August 2026  
**Classification:** **A Fail** · **B Store-works**  
**Run:** `runs/2026-08-14_213855_tm066`

Recipe jump: **correct the dirty English store.** Same concurrent-bind multi-rare W as TM.0.6.5. Once S names here, stop committing. After a real stamp, drop pages that never got an act. TM.0.6.5 train S had 11 files (only `p99` bound). This slice: train S is **1 file**, `bind=push`, n_revised=3, probe **PRESS** / foil C **HOLD**. A later C life TUNEs, but by binding clutter hapax `xenon` on `c08`, not `adjust` on `p98`. Wipe-between loses A. Cortex frozen (`a485b26b…`). `n_forced=0`. `n_train` was not raised. Revise / here-only default **off**.

Skipped on purpose: dropping `has_code`, removing `domain=`, math, solving B (B happened to pass the motor bar; not retuned), restoring unique-rare, adding a ranker, raising `MAX_TRAIN_S_FILES`.

## Question

Can a growing English store stop stamp-collecting unmarked clutter, or does every never-wipe life leave a junk drawer beside the fact?

## Predeclared

| ID | A | B |
|----|---|----|
| Confound | Cortex moves; unique-rare restored; `push` in the agent; drop `has_code` / `domain=`; stamp-new-here off; argon as clutter hapax; n_train raised | same |
| Fail | Untrained PRESS; train S not PRESS; train S still >4 files; C life loses A or misses TUNE; wipe-between still PRESS; bind=`xenon`/`neon`/`krypton`/`argon`; nonce PRESS | Untrained PRESS; A miss; C miss; n>4; nonce PRESS |
| Store-works | Multi-rare English W; train S small; A PRESS / C HOLD from `push`; C life A PRESS / C TUNE from `adjust`; wipe-between A HOLD; nonce HOLD | Same without splitting the return |

## Headline

| Check | A split | B shared return |
|-------|---------|-----------------|
| Classification | **Fail** | **Store-works** |
| Untrained A / foil C | `hold` / `hold` | `hold` / `hold` |
| After train, S: A / foil C | **`press` / `hold`** | **`press` / `hold`** |
| C life on that S: A / C | `press` / `tune` | **`press` / `tune`** |
| Wipe-between A | **`hold`** | **`hold`** |
| Train S n files | **1** | **1** |
| n_revised train | 3 | 18 |
| Train S binds | **`push`** | `krypton` |
| C life binds | `push` + **`xenon`** | `krypton` + `xenon` |
| Train last 50 | 1.00 | 1.00 |
| Cortex | unchanged | unchanged |

A train note is only `p99`: `bind=push`, `did=press`, `argon` kept. C added `c08` `bind=xenon` `did=tune`. Closed clutter-only HOLD. Nonce HOLD.

## Compare

**A** is the jump: TM.0.6.5’s eleven unmarked files are not required. Here-only + sweep leave one inspectable A note that still PRESS. The English two-station bar fails: a new station still takes the first leftover rare in W (`xenon`), not `p98`/`adjust`. New-here stays rare-only (a common in-hand page is not a CS). Do not restore the junk drawer. Do not rank `p98`.

**B** shared return **Store-works** on the motor bar (n=1, PRESS/TUNE, last-50 1.00). First CS was `krypton`; C bound `xenon`. Not the jump. Not retuned.

## Audit (not retuned)

- TM.0.5.0–0.6.5 unit tests still pass.
- Default door agent: revise / here-only / block-here / stamp-new-here / one-bind **off**.
- TM.0.6.0–0.6.5 `make()` leave revise/here-only off (except TM.0.5.9). TM.0.6.5 A remains Store-works without them; its train S stays large.
- Search still computes `has_code`. `domain="dial"` still set. No English lexicon in the agent.

## Honest limits

- Stream-first is still frozen grammar, not English syntax.
- Tiny closed corpus, not Wikipedia.
- Search is still `{has_code, has_rare}`. New-here fallback is still first rare in W order.
- Split credit still teaches A to attend `push`; C life has no such signal for `adjust`.
- Shared return passing here does not rewrite this Fail or earlier Fail slices.
- Still five acts, innate names, `domain=`. Math is a later life.

## Reproduce

```bash
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
python -m experiments.run_tm066
```
