# v16 results: ok= vs newest, and shared return

**Date:** 14 August 2026  
**Classification:** **A Store-works** · **B Fail**  
**Run:** `runs/2026-08-14_094047_v16`

## Question

Two leftover claims after v15:

- **A.** Newest `when=` can be **wrong**. Prefer a success-marked note (`ok=1`) over recency.
- **B.** v15 used **split** credit. Does **one shared return** still compose?

Same frozen cortex. Generic copy. `n_forced=0`. Probe greedy. W has no answers. No door/`action=` in rank features. Do not restore newest-wins or split credit to rescue an arm.

## Predeclared

### A — rank `{is_newest, has_ok}`

Untrained recency prior (`w_rank = [1.2, 0]`). Life authors `ok=1` on a complete success write. Then a **newer junk** note is planted. Train on red. Held-out green (junk `open`). Trained path has no `force_write`. Untrained newest readout uses `force_use` only so copy is on (otherwise the use-gate hides recency).

| ID | If |
|----|----|
| Confound | Cortex moves; disable-S `use_key`; answer in W; door/`action=` in rank features; probe explores |
| Fail | Untrained already `use_key`; red works, green doesn’t; newest-wins still solves red |
| Store-works | Untrained follows newest (`wait`); trained red `use_key` from `ok=1`; green `wait`; newest-prior control still `wait` |

### B — shared return (v15 setup)

One scalar `r = probe correct` updates write, schema, use, and pick together. Same stale-note red train as v15.

| ID | If |
|----|----|
| Confound | Cortex moves; disable-S `use_key`; answer in W; probe explores |
| Fail | Red still `open`; green fails; last-50 ≈ 0 |
| Store-works | Same as v15: trained red `use_key`; green `wait`; apply-all mixes |

## Headline

| Check | A ok= vs newest | B shared return |
|-------|-----------------|-----------------|
| Classification | **Store-works** | **Fail** |
| Untrained red | `wait` (newest junk) | `open` (use-gate off) |
| Trained red | **`use_key`** (`has_ok=True`) | `open` |
| Held-out green | **`wait`** | `open` |
| Control | newest-prior still `wait` | apply-all `open` |
| Train last 50 | 0.84 | **0.00** |
| Cortex | unchanged | unchanged |

## Compare

**A** is a new retrieve claim: recency is not enough once junk is newer. The older `ok=1` file still has the integer; the head only learned which mark to trust.

**B** is a training claim: v15’s split credit was load-bearing. Shared return moved all four heads (hashes changed) but never solved the probe. Last-50 stayed 0. Do not treat split credit as a silent clamp on A; A still split write/schema vs use/rank. B is the arm that forbids that split.

## Honest limits

- `ok=1` is genome (written on a successful complete note), not a learned fact.
- Untrained A newest readout still needs copy on (`force_use`) or the use-gate hides recency as `open`.
- Rank is two features, two files. Not a general ranker.
- B was not retuned. A Fail here is the result.

## Reproduce

```bash
python tests/test_v16.py
python tests/test_v15.py
python -m experiments.run_v16
```
