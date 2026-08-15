# TM.0.9.0 results: first math life / motor bar

**Date:** 15 August 2026  
**Classification:** **A Fail** · **B Store-works**  
**Run:** `runs/2026-08-14_235840_tm090`

Recipe jump: **count unread rares in search.** Same one-machine one-return English store as TM.0.8.2. Genome may count how many rare tokens a page still adds that S lacks. It may not add. No `+` in cortex. Retrieve used `xenon`. C life retrieve used `neon`. Motors PRESS/TUNE. Cortex frozen (`a485b26b…`). `n_forced=0`. `n_train` was not raised.

## Question

Can a later life use the cardinality of a stream already in view, or does search still need a place-int `has_code` bit — and does counting prefer `push`?

## Headline

| Check | A count-search | B motor bar |
|-------|----------------|-------------|
| Classification | **Fail** | **Store-works** |
| Untrained A / foil C | `hold` / `hold` | `hold` / `hold` |
| After train, S: A / foil C | **`press` / `hold`** | **`press` / `hold`** |
| C life on that S: A / C | `press` / `tune` | **`press` / `tune`** |
| Used bind A / C | **`xenon`** / **`neon`** | — |
| Train S n files | **2** | **2** |
| Train last 50 | 0.90 | 0.94 |

## Compare

**A** is the jump: search’s first feature is “this page still adds a rare token S lacks,” not “this page carries the place int.” Many clutter pages also add rares. Count did not prefer `push`. Do not put `+` in cortex or restore a unique pair to rescue retrieve.

**B** motor bar **Store-works** (n=2, last-50 0.94). Not the English bar. Not retuned.

## Honest limits

- Cardinality is a legal genome skill. It does not teach which English word is the act.
- Stream-first bind, five acts, one machine. More math later, after S holds the right language.

## Reproduce

```bash
python tests/test_tm090.py
python tests/test_tm082.py
python tests/test_tm050.py
python -m experiments.run_tm090
```
