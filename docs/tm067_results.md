# TM.0.6.7 results: in-hand new-here / shared return

**Date:** 14 August 2026  
**Classification:** **A Fail** · **B Store-works**  
**Run:** `runs/2026-08-14_215734_tm067`

Recipe jump: **a new station stamps the attended rare page, not the first leftover rare in W.** Same corrected concurrent-bind multi-rare English W as TM.0.6.6. Train S is **1 file**, `bind=push`, n_revised=3, probe **PRESS** / foil C **HOLD**. A later C life TUNEs, but by binding clutter hapax `neon` on `c09` (the page search held), not `adjust` on `p98`. Leftover W-order walk is off. Wipe-between loses A. Cortex frozen (`a485b26b…`). `n_forced=0`. `n_train` was not raised. In-hand new-here default **off**.

Skipped on purpose: dropping `has_code`, removing `domain=`, math, solving B (B happened to pass the motor bar; not retuned), restoring unique-rare, adding a ranker, restoring the junk drawer or leftover walk, raising `MAX_TRAIN_S_FILES`.

## Question

Can a growing English store take a second station as the page in hand, or does every new place still stamp whatever leftover rare sits in the unread pile?

## Predeclared

| ID | A | B |
|----|---|----|
| Confound | Cortex moves; unique-rare restored; `push` in the agent; drop `has_code` / `domain=`; leftover walk kept; ranker for `p98`; argon as clutter hapax; n_train raised | same |
| Fail | Untrained PRESS; train S not PRESS; train S still >4 files; C life loses A or misses TUNE; wipe-between still PRESS; bind=`xenon`/`neon`/`krypton`/`argon`; nonce PRESS | Untrained PRESS; A miss; C miss; n>4; nonce PRESS |
| Store-works | Multi-rare English W; train S small; A PRESS / C HOLD from `push`; C life A PRESS / C TUNE from `adjust` on the attended page; wipe-between A HOLD; nonce HOLD | Same without splitting the return |

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
| C life binds | `push` + **`neon`** | `krypton` + `neon` |
| Train last 50 | 1.00 | 1.00 |
| Cortex | unchanged | unchanged |

A train note is only `p99`: `bind=push`, `did=press`, `argon` kept. C added `c09` `bind=neon` `did=tune`. Closed clutter-only HOLD. Nonce HOLD.

## Compare

**A** is the jump: TM.0.6.6’s leftover walk (`c08` / `xenon` in W order) is not required. New-here with a common page in hand stamps nothing. New-here with a rare page in hand stamps that page. The English two-station bar fails: C search held `c09` / `neon`, not `p98` / `adjust`. Do not restore the leftover walk. Do not rank `p98`.

**B** shared return **Store-works** on the motor bar (n=1, PRESS/TUNE, last-50 1.00). First CS was `krypton`; C bound `neon`. Not the jump. Not retuned.

## Audit (not retuned)

- TM.0.5.0–0.6.6 unit tests still pass.
- Default door agent: in-hand new-here / revise / here-only / block-here / stamp-new-here / one-bind **off**.
- TM.0.6.0–0.6.6 `make()` leave in-hand new-here off. TM.0.6.6 A remains Fail without it; its C life still takes leftover `xenon`.
- Search still computes `has_code`. `domain="dial"` still set. No English lexicon in the agent.

## Honest limits

- Stream-first is still frozen grammar, not English syntax.
- Tiny closed corpus, not Wikipedia.
- Search is still `{has_code, has_rare}`. New-here no longer walks W; it still stamps whatever rare page find attended.
- Split credit still teaches A to attend `push`; C life has no such signal for `adjust`.
- Shared return passing here does not rewrite this Fail or earlier Fail slices.
- Still five acts, innate names, `domain=`. Math is a later life.

## Reproduce

```bash
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
python -m experiments.run_tm067
```
