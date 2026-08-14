# TM.0.8.0 results: scale English Open W / shared return

**Date:** 15 August 2026  
**Classification:** **A Fail** · **B Store-works**  
**Run:** `runs/2026-08-14_231534_tm080`

Recipe jump: **scale of English Open W.** Clutter is 64 distinct multi-paragraph English pages, not 11. Sixteen of them also have three hapax, so `p99`/`p98` are not a unique novel pair. Same keep-steerer local-alias recipe as TM.0.7.2. Retrieve used `xenon`. C life retrieve used `krypton`. Motors PRESS/TUNE. Cortex frozen (`a485b26b…`). `n_forced=0`. `n_train` was not raised.

## Question

Does the same English find/stamp/keep-steerer recipe still work when unread W is a pile of documents, or does it need a dozen-page toy library?

## Headline

| Check | A split | B shared return |
|-------|---------|-----------------|
| Classification | **Fail** | **Store-works** |
| Untrained A / foil C | `hold` / `hold` | `hold` / `hold` |
| After train, S: A / foil C | **`press` / `hold`** | **`press` / `hold`** |
| C life on that S: A / C | `press` / `tune` | **`press` / `tune`** |
| Used bind A / C | **`xenon`** / **`krypton`** | — |
| Distinct clutter | **64** | **64** |
| Two-rare clutter | **16** | **16** |
| Train S n files | **2** | **2** |
| Train last 50 | 0.94 | 0.94 |

## Compare

**A** is the jump: a 64-page English pile, more two-rare clutter, same `{has_code, has_rare}` / keep-steerer recipe. Search still retrieved `c08` / `xenon`. Do not restore a unique pair or a `p98` ranker to rescue retrieve.

**B** motor bar **Store-works** (n=2, last-50 0.94). Not the jump. Not retuned.

## Honest limits

- Scale does not teach search to prefer `push`. Keep-steerer still cements the first successful clutter note.
- Stream-first bind, five acts, `domain=`. Math is a later life.

## Reproduce

```bash
python tests/test_tm080.py
python tests/test_tm072.py
python tests/test_tm050.py
python -m experiments.run_tm080
```
