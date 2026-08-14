# TM.0.6.8 results: find-novel / shared return

**Date:** 14 August 2026  
**Classification:** **A Store-works** · **B Store-works**  
**Run:** `runs/2026-08-14_220921_tm068`

Recipe jump: **find the unread page that would add the most rare tokens S does not already have, and attend it without stamp-collecting leftover hapax.** Same in-hand new-here English store as TM.0.6.7. Train S is **1 file**, `bind=push`, n_revised=0, probe **PRESS** / foil C **HOLD**. A later C life TUNEs from `p98` `bind=adjust`, not from `xenon`/`neon`. Wipe-between loses A. Cortex frozen (`a485b26b…`). `n_forced=0`. `n_train` was not raised. Find-novel default **off**.

Skipped on purpose: dropping `has_code`, removing `domain=`, math, ranking `p98` by name, restoring unique-rare, restoring leftover W walk, raising `MAX_TRAIN_S_FILES`. B happened to pass the two-station bar; not retuned.

## Question

Can search prefer an unread page that would teach S more rare tokens, or does every new station still attend a leftover hapax?

## Predeclared

| ID | A | B |
|----|---|----|
| Confound | Cortex moves; unique-rare restored; `push` in the agent; drop `has_code` / `domain=`; leftover walk kept; ranker for `p98`; argon as clutter hapax; n_train raised | same |
| Fail | Untrained PRESS; train S not PRESS; train S still >4 files; C life loses A or misses TUNE; wipe-between still PRESS; bind=`xenon`/`neon`/`krypton`/`argon`; nonce PRESS | Untrained PRESS; A miss; C miss; n>4; nonce PRESS |
| Store-works | Multi-rare English W; train S small; A PRESS / C HOLD from `push`; C life A PRESS / C TUNE from `adjust`; wipe-between A HOLD; nonce HOLD | Same without splitting the return |

## Headline

| Check | A split | B shared return |
|-------|---------|-----------------|
| Classification | **Store-works** | **Store-works** |
| Untrained A / foil C | `hold` / `hold` | `hold` / `hold` |
| After train, S: A / foil C | **`press` / `hold`** | **`press` / `hold`** |
| C life on that S: A / C | **`press` / `tune`** | **`press` / `tune`** |
| Wipe-between A | **`hold`** | **`hold`** |
| Train S n files | **1** | **1** |
| n_revised train | 0 | 0 |
| Train S binds | **`push`** | **`push`** |
| C life binds | **`push` + `adjust`** | **`push` + `adjust`** |
| Train last 50 | 1.00 | 0.94 |
| Cortex | unchanged | unchanged |

A train note is only `p99`: `bind=push`, `did=press`, `argon` kept. C added `p98` `bind=adjust` `did=tune`. Closed clutter-only HOLD. Nonce HOLD.

## Compare

**A** is the jump: TM.0.6.7’s in-hand stamp still bound leftover `neon` because search attended a one-hapax page and collect kept copying W into S. Find-novel keeps unread pages with the most rare tokens not already in S (`p99`/`p98` have two; clutter hapax have one) and attends that page without stamp-collecting the rest. C life TUNEs from `adjust`. Do not rank `p98` by name. Do not restore unique-rare.

**B** shared return **Store-works** on this slice (n=1, PRESS/TUNE from `push`+`adjust`, last-50 0.94). Not the jump. Not retuned.

## Audit (not retuned)

- TM.0.5.0–0.6.7 unit tests still pass.
- Default door agent: find-novel / in-hand new-here / revise / here-only / block-here / stamp-new-here / one-bind **off**.
- TM.0.6.0–0.6.7 `make()` leave find-novel off. TM.0.6.7 A remains Fail without it; its C life still bound `neon`.
- Search still computes `has_code`. `domain="dial"` still set. No English lexicon in the agent.

## Honest limits

- Stream-first is still frozen grammar, not English syntax.
- Tiny closed corpus, not Wikipedia.
- Search is still `{has_code, has_rare}` after the novel-count filter. A clutter page with two hapax would compete with `p98`. TM.0.6.9 did that (three body hapax, because `# p99`/`# p98` already add rare `p`); A Fail.
- Split credit still teaches A to attend `push`; C life has no split `r_find` for `adjust`. Find-novel is frozen grammar, not that signal.
- Shared return passing here does not rewrite earlier Fail slices.
- Still five acts, innate names, `domain=`. Math is a later life.

## Reproduce

```bash
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
python -m experiments.run_tm068
```
