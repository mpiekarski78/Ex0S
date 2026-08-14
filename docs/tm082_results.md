# TM.0.8.2 results: one machine / motor bar

**Date:** 15 August 2026  
**Classification:** **A Fail** · **B Store-works**  
**Run:** `runs/2026-08-14_234508_tm082`

Recipe jump: **no `domain=` switch.** Motors, affordances, and station names come from body size (`n_actions=5`) and the current percept. Same one-return 64-page English keep-steerer store as TM.0.8.1. Retrieve used `xenon`. C life retrieve used `neon`. Motors PRESS/TUNE. Cortex frozen (`a485b26b…`). `n_forced=0`. `n_train` was not raised.

## Question

Is there one machine whose body is the act count and the percept, or does DNA still switch on a dial/door label?

## Headline

| Check | A one machine | B motor bar |
|-------|---------------|-------------|
| Classification | **Fail** | **Store-works** |
| Untrained A / foil C | `hold` / `hold` | `hold` / `hold` |
| After train, S: A / foil C | **`press` / `hold`** | **`press` / `hold`** |
| C life on that S: A / C | `press` / `tune` | **`press` / `tune`** |
| Used bind A / C | **`xenon`** / **`neon`** | — |
| n_actions | **5** | **5** |
| Train S n files | **2** | **2** |
| Train last 50 | 0.90 | 0.94 |

## Compare

**A** is the jump: `if self.domain ==` is gone. Search still retrieved `xenon`. One machine did not prefer `push`. Do not restore a domain label to rescue retrieve.

**B** motor bar **Store-works** (n=2, last-50 0.94). Not the English bar. Not retuned.

## Honest limits

- One machine is the honest body. It does not teach search to prefer `push`.
- Stream-first bind, five acts, `{has_code, has_rare}`. Math is a later life.

## Reproduce

```bash
python tests/test_tm082.py
python tests/test_tm081.py
python tests/test_tm050.py
python -m experiments.run_tm082
```
