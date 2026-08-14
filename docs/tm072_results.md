# TM.0.7.2 results: keep-steerer / shared return

**Date:** 15 August 2026  
**Classification:** **A Fail** · **B Store-works**  
**Run:** `runs/2026-08-14_231051_tm072`

Recipe jump: **after a successful act, keep the retrieved/in-hand note at this station and drop the rest.** Same local-alias English store as TM.0.7.1. Search still retrieved `c08` / `xenon`. Keep-steerer then dropped `p99` / `push`. C life retrieve used `neon`. Motors PRESS/TUNE. Cortex frozen (`a485b26b…`). `n_forced=0`. Keep-steerer default **off**.

## Question

Can S keep the file that steered, or does the first successful clutter note become the station’s only remaining English?

## Headline

| Check | A split | B shared return |
|-------|---------|-----------------|
| Classification | **Fail** | **Store-works** |
| Untrained A / foil C | `hold` / `hold` | `hold` / `hold` |
| After train, S: A / foil C | **`press` / `hold`** | **`press` / `hold`** |
| C life on that S: A / C | `press` / `tune` | **`press` / `tune`** |
| Used bind A / C | **`xenon`** / **`neon`** | — |
| Train S n files | **2** | **2** |
| Train last 50 | 0.86 | 0.94 |

A train S: `c08` `bind=xenon` and `c10` `bind=krypton` (both PRESS at A). `push` is gone. C life added `c09` `bind=neon` (TUNE).

## Compare

**A** is the jump: after success, other same-here notes are dropped. Search still picked clutter first. Keep-steerer cemented that file. Do not restore a unique two-rare pair, a `p98` ranker, or newest-among-filename-order to rescue retrieve.

**B** motor bar **Store-works** (n=2, last-50 0.94). Not the jump. Not retuned.

## Honest limits

- Keep-steerer cannot prefer `push` if search never retrieves `p99`. It keeps the first successful retrieve plus the just-stamped in-hand page.
- Retrieve among same-here notes is still `{has_code, has_rare}` / file order. `c08` wins at A.
- Stream-first bind, five acts, `domain=`. Math is a later life.

## Reproduce

```bash
python tests/test_tm072.py
python tests/test_tm071.py
python tests/test_tm050.py
python -m experiments.run_tm072
```
