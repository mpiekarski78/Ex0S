# TM.0.7.1 results: local-alias / shared return

**Date:** 14 August 2026  
**Classification:** **A Fail** · **B Store-works**  
**Run:** `runs/2026-08-14_230532_tm071`

Recipe jump: **alias a page word to a motor only from the retrieved note.** Same retry-novel dirty English store as TM.0.7.0 (n=4). A global bind→motor table was a growing English lexicon. Retrieve used `xenon`, not `push`. C life retrieve used `adjust`. Motors PRESS/TUNE. Cortex frozen (`a485b26b…`). `n_forced=0`. Local-alias default **off**.

## Question

Can use be the retrieved file’s bind→did, or does every stamp in S become a species synonym?

## Headline

| Check | A split | B shared return |
|-------|---------|-----------------|
| Classification | **Fail** | **Store-works** |
| Untrained A / foil C | `hold` / `hold` | `hold` / `hold` |
| After train, S: A / foil C | **`press` / `hold`** | **`press` / `hold`** |
| C life on that S: A / C | `press` / `tune` | **`press` / `tune`** |
| Used bind A / C | **`xenon`** / `adjust` | — |
| Train S n files | **4** | **4** |
| Train last 50 | 0.96 | 0.96 |

## Compare

**A** is the jump: bind→did is file-local. Search still retrieved `c08` / `xenon` at A. C retrieved `adjust`. Dirty S may keep clutter; the English bar is which file was used. Do not restore a global hapax lexicon.

**B** motor bar **Store-works** (n=4, last-50 0.96). Not the jump. Not retuned.

## Honest limits

- Retrieve among same-here notes is still `{has_code, has_rare}` / file order. `c08` wins at A.
- Stream-first bind, five acts, `domain=`. Math is a later life.

## Reproduce

```bash
python tests/test_tm071.py
python tests/test_tm070.py
python tests/test_tm050.py
python -m experiments.run_tm071
```
