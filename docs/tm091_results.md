# TM.0.9.1 results: keep untested competing hypotheses / motor bar

**Date:** 15 August 2026  
**Classification:** **A Store-works** · **B Fail**  
**Run:** `runs/2026-08-15_091638_tm091`

Recipe jump: **hypothesis survival.** Same one-machine count-search English store as TM.0.9.0. Success of one same-here bind does not delete an untested rival. Notes carry `hyp=untried` / `supported` / `contradicted` on inspectable S. Retrieve prefers untried / least-tried. Retrieve used `thallium`. C life used `adjust`. Motors PRESS/TUNE. Cortex frozen (`a485b26b…`). `n_forced=0`. `n_train` was not raised. `MAX_TRAIN_S_FILES` was not raised.

## Question

If `xenon → press → success` and `push → press → success` return the same observation, can the recipe keep both hypotheses — or does keep-steerer still delete the untested rival because the first file worked?

## Headline

| Check | A hyp-survive | B motor bar |
|-------|---------------|-------------|
| Classification | **Store-works** | **Fail** |
| Untrained A / foil C | `hold` / `hold` | `hold` / `hold` |
| After train, S: A / foil C | **`press` / `hold`** | **`press` / `hold`** |
| C life on that S: A / C | `press` / `tune` | `press` / `tune` |
| Used bind A / C | **`thallium`** / **`adjust`** | — |
| Train S n files | **16** | **17** |
| hyp untried / supported / contradicted | 0 / **16** / 0 | 0 / **17** / 0 |
| Train last 50 | 0.96 | 0.98 |

S after A train still has `p99` (`bind=push`, `hyp=supported`) and `c08` (`bind=xenon`, `hyp=supported`), plus fourteen other same-here clutter binds, all supported. Keep-steerer no longer deletes the rival after the first success.

## Compare

**A** is the jump: evidence about one bind does not remove an untested (or equally supported) other. Sixteen same-here notes are all `supported`. The world gave the same observation for each. No honest ranking rule can discover that `push` is the meaningful English token. Retrieve used `thallium`. That is allowed. The experimenter bar `used_bind == "push"` is a lexical prior; 0.9.1 does not require it.

**B** motors still PRESS/TUNE (last-50 0.98). n=17 is stamp-collecting. **Fail**. The cap stays 4. Not retuned.

## Honest limits

- Survival is not meaning. `push` and `xenon` (and `thallium`, `neon`, …) are equally supported on this world.
- Keeping every successful same-here note grows S. That is why B Fails. Do not raise `MAX_TRAIN_S_FILES`.
- MATCH landed in TM.0.9.2. Next is EVIDENCE (TM.0.9.3): among applicable rivals, prefer the better-supported one. Keep hyp-survive.

## Reproduce

```bash
python tests/test_tm091.py
python tests/test_tm090.py
python tests/test_tm050.py
python -m experiments.run_tm091
```
