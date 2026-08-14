# TM.0.8.1 results: one return is the recipe / motor bar

**Date:** 15 August 2026  
**Classification:** **A Fail** · **B Store-works**  
**Run:** `runs/2026-08-14_233053_tm081`

Recipe jump: **one shared return is the A recipe.** Same 64-page English keep-steerer store as TM.0.8.0. A no longer gets split find/mark/use credit. One signal: the probe worked after the life. Retrieve used `xenon`. C life retrieve used `neon`. Motors PRESS/TUNE. Cortex frozen (`a485b26b…`). `n_forced=0`. `n_train` was not raised. `trained_split` is **false** on both arms.

## Question

Can look and it-worked share one life signal, or does the recipe still need an experimenter who scores find, mark, and use apart?

## Headline

| Check | A one return | B one-return motor bar |
|-------|--------------|------------------------|
| Classification | **Fail** | **Store-works** |
| trained_split | **false** | **false** |
| Untrained A / foil C | `hold` / `hold` | `hold` / `hold` |
| After train, S: A / foil C | **`press` / `hold`** | **`press` / `hold`** |
| C life on that S: A / C | `press` / `tune` | **`press` / `tune`** |
| Used bind A / C | **`xenon`** / **`neon`** | — |
| Train S n files | **2** | **2** |
| Train last 50 | 0.90 | 0.94 |

## Compare

**A** is the jump: split credit is gone. Search still retrieved `c08` / `xenon`. One return did not prefer `push`. Do not restore split find/mark/use to rescue retrieve.

**B** motor bar **Store-works** (n=2, last-50 0.94). Not the English bar. Not retuned.

## Honest limits

- One return is the honest recipe. It does not teach search to prefer `push`.
- Stream-first bind, five acts, `domain=`. Math is a later life.

## Reproduce

```bash
python tests/test_tm081.py
python tests/test_tm080.py
python tests/test_tm050.py
python -m experiments.run_tm081
```
