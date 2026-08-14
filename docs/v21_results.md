# v21 results: first-file vs dump-all among unread W hits

**Date:** 14 August 2026  
**Classification:** **A Store-works** · **B Store-works**  
**Run:** `runs/2026-08-14_103504_v21`

## Question

v20 found a page by learning the query name `here=`. Collect still kept **`w_hits[0]`** (sorted filename). A real shelf has many pages on the same index.

v21 freezes the v20 query (`place_key="here"`) and the v5 commit rule. Copy is frozen on (`force_use`). The boxed head learns **which matching unread page to keep**. Frozen WHAT when the head is on: newest `when=`. No door id, no `action=` in the head. No `d0.tag`. No writes from life. Cortex frozen. Probe greedy.

- **A.** Untrained keeps filename-first: `aaa.tag` `{here:0, action:0, when:1}` → `wait`. Useful `p99.tag` is newer (`when:9`).
- **B.** Untrained dumps **every** `here=` hit into S. Mix copies `wait` and `use_key`; red probe `wait`.

Held-out green: `aag.tag` `{here:2, action:1, when:1}` (`open`) vs `p98.tag` `{here:2, action:0, when:9}` (`wait`). Recency-swap control: newest is junk `wait`; must not still `use_key` (that would be memorizing `p99.tag`).

## Predeclared

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; disable-S `use_key`; `d0`/`d2` in W; empty-S green `wait`; writes from life | same |
| Fail | Untrained already `use_key` or already newest; green fails; freeze-newest; recency-swap still `use_key` | Untrained dump already `use_key`; S still has both pages; recency-swap `use_key` |
| Store-works | Untrained `wait` (`aaa.tag`); unmount W red `use_key` from `p99.tag`; green `wait`; first-file and recency-swap stay `wait` | Untrained `wait` (both files); trained S is `p99.tag` only; green `wait`; dump and recency-swap stay `wait` |

Do not freeze-newest or restore filename-first / dump-all / `d0.tag` to rescue a plot.

## Headline

| Check | A first vs newest | B dump vs newest |
|-------|-------------------|------------------|
| Classification | **Store-works** | **Store-works** |
| Untrained | `wait` (`aaa.tag` only) | `wait` (`aaa.tag`+`p99.tag`) |
| Trained red, unmount W | **`use_key`** (`p99.tag` `when=9`) | **`use_key`** (`p99.tag` only) |
| Held-out green, unmount W | **`wait`** (`p98.tag`) | **`wait`** (`p98.tag`) |
| First/dump control | `wait` | `wait` |
| Recency swap (newest is junk) | `wait` | `wait` |
| Empty S / disable-S | `open` | `open` |
| Train last 50 | 0.96 | 0.98 |
| Cortex | unchanged | unchanged |

## Compare

Both arms learned the same frozen pick (newest `when=`) from different wrong priors: A from **one wrong file**, B from **every matching file**. Recency-swap stays `wait`, so they did not memorize `p99.tag`. Green used a page that was never in train W, with junk `open` on the same `here=2` index.

## Honest limits

- Query name is frozen to `here=` (v20). Copy is frozen on. Collect is frozen commit-on-hit.
- Newest `when=` is still genome. The experimenter planted a higher `when=` on the useful page. This is not relevance search, not TF-IDF, not Wikipedia.
- Exact tag match. Two-key world. Four acts. Split credit not required (one head).
- `w_hits[0]` remains the default when `use_wsel_head` is off (v20 unchanged).

## Reproduce

```bash
python tests/test_v21.py
python tests/test_v20.py
python tests/test_v8.py
python -m experiments.run_v21
```
