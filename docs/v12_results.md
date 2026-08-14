# v12 results: learn select vs dump

**Date:** 14 August 2026  
**Classification:** **Store-works**  
**Run:** `runs/2026-08-14_091402_v12`

## Question

> Can a boxed policy learn **not to dump S**, without door identity in its features, while a held-out third life still works?

v11 hardcoded `retrieve_policy="select"`. v12: a retrieve head chooses **select** vs **dump** from `{n_store ≥ 2, n_hits ≥ 1}` only. Cortex frozen. Motor act still from `action=` in the file. W has no `d0`/`d1`/`d2`.

## Predeclared

| ID | If |
|----|----|
| Confound | Cortex hash moves; disable-S still `use_key`; answer in W; `n_forced > 0`; probe explores; policy emits `use_key` |
| Fail | Retrieve head unchanged; untrained already solves red; trained still dumps; blue missing/wrong; dump-all as good as select on all probes |
| Store-works | Retrieve head changed; untrained dumps (red `wait`); trained selects (red `use_key`, green `wait`); held-out blue `open`; dump-all still fails red |

Train retrieve on red+green S only. Then a **blue** free life authors `d1.tag` `{door:1, action:1}` — not used to train the head.

## Headline

| Check | Result |
|-------|--------|
| Untrained retrieve, red | **dump**, `wait` |
| Trained retrieve, red | **select**, **`use_key`** |
| Trained retrieve, green | **select**, **`wait`** |
| Dump-all red (control) | `wait` |
| Held-out blue file | `d1.tag` authored into S with `d0`+`d2` |
| Trained retrieve, blue | **select**, **`open`** |
| Dump-all blue | `wait` |
| Empty S / disable-S | `open` |
| Cortex SHA256 | unchanged |
| Retrieve head | changed |
| Retrieve train last 50 | 0.90 |

The head did not learn “red → use_key”. Features are pile-size and match-count. Blue `open` is what `d1.tag` says.

## What this means

v11: pick (frozen).  
v12: **learn that dumping is not using.**

## Honest limits

- Binary select vs dump, two features. Not a learned ranker among many matches.
- Tag match on `door=` is still the frozen matcher once select is chosen.
- Three files, not a wiki.

## Reproduce

```bash
python tests/test_v12.py
python tests/test_v11.py
python -m experiments.run_v12
```
