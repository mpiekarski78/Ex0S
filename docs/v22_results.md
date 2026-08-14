# v22 results: complete vs stub / joint find+pick+use

**Date:** 14 August 2026  
**Classification:** **A Store-works** · **B Store-works**  
**Run:** `runs/2026-08-14_103949_v22`

## Question

v21 picked among unread W hits by planted recency (`when=9` on the useful page). Newest is genome.

v22 takes that cheat out on **A**, and takes the v20/v21 clamps out on **B**. Cortex frozen. Generic copy. Exact match. No `d0.tag`. No writes from life. Probe greedy.

- **A.** No `when=`. Junk is a stub `{here:0}` (`aaa.tag`, sorts first). Useful is `{here:0, action:2}`. Query and copy frozen. Head: keep the page that has `action=` / `do=`. Features `{has_payload, n_hits≥2}` — no door id, no integer.
- **B.** Joint: match + wsel + use-gate, no `force_use`, no frozen `here=`. W has v21’s first/newest `here=` pages **and** v20’s `door=` junk. Split credit (found `here=` / kept `action=2` / unmount probe).

Held-out green: A stub vs complete `wait`; B first `open`, newest `wait`, `door=` junk `open`.

## Predeclared

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; disable-S `use_key`; `d0`/`d2` in W; planted `when=`; empty-S green `wait` | Cortex moves; disable-S `use_key`; `d0`/`d2` in W; empty-S green `wait` |
| Fail | Untrained already `use_key` or already complete; green fails; complete-is-junk still `use_key`; stub-only `use_key` | Untrained already `use_key`; green fails; freeze-match `door=` / freeze-first / use-off still `use_key`; a head frozen off |
| Store-works | Untrained `open` (stub); unmount red `use_key` from complete `p99.tag`; green `wait`; swap and stub-only fail | Untrained `open`; red `use_key` from newest `here=` page; green `wait`; door=/first/use-off fail |

Do not plant `when=` on A or freeze `here=` / `force_use` on B to rescue a plot.

## Headline

| Check | A complete vs stub | B joint no clamps |
|-------|--------------------|-------------------|
| Classification | **Store-works** | **Store-works** |
| Untrained | `open` (`aaa.tag` stub) | `open` (use off / `door=`) |
| Trained red, unmount W | **`use_key`** (`p99.tag`, no `when=`) | **`use_key`** (`p99.tag` `when=9`) |
| Held-out green, unmount W | **`wait`** (`p98.tag`) | **`wait`** (`p98.tag`) |
| Complete-is-junk / stub-only | `wait` / `open` | n/a |
| Match `door=` / first-file / use-off | n/a | `wait` / `wait` / `open` |
| Train last 50 | 1.00 | 0.92 |
| Cortex | unchanged | unchanged |

## Compare

A does not need recency: a stub has no payload, an article does. Swap (complete file is `wait`) stays `wait`, so they did not memorize `p99.tag`.

B shows v20 find and v21 pick can run together with the use-gate, under split credit. Each clamp-off control fails, so all three heads are load-bearing.

## Honest limits

- A’s complete-rule is still genome (first hit that has `action=` / `do=`). The head only learns *when* to use it vs filename-first.
- B still uses planted `when=` for newest and a two-key match menu. Split credit; shared return was not restored.
- Exact match. Four acts. Not search, not English, not Wikipedia.

## Reproduce

```bash
python tests/test_v22.py
python tests/test_v21.py
python tests/test_v20.py
python -m experiments.run_v22
```
