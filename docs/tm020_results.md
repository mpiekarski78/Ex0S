# TM.0.2.0 results: scale of W / shared return

**Date:** 14 August 2026  
**Classification:** **A Store-works** · **B Store-works**  
**Run:** `runs/2026-08-14_111943_tm020`

Same messy search as TM.0.1.2 (`{has_code, has_rare}`), same door world, same useful page `{where:0, action:2, pad:7}`. W grows from ~12 files to **256**. Exact `loc=` / `door=` / `here=` still miss. Copy stays frozen `action=`. The agent source has no `where` string. Search features were not changed to rescue scale.

## Question

TM.0.1.2 ranked a handful of unread files. TM.0.2.0 asks whether the same boxed head still commits the useful page when W is hundreds of messy `.tag` files — the first wiki-shaped **count**, not English Wikipedia.

Untrained prior: any file that carries the door code (first on ties → early clutter `c000.tag`… with `place=0`). Cortex frozen. Generic copy. No `d0.tag`. No `when=`. No writes from life. Probe greedy. `n_train=10000` (predeclared look-count match to TM.0.1.2’s ~10 random hits on ~12 files; not raised after the plot).

- **A.** Split: found `action=2` in S / unmount probe.
- **B.** One shared return: `r` = unmount probe correct, applied to search and use.

Held-out green: same 256-scale clutter + `p98.tag` `{where:2, action:0, pad:7}`.

## Predeclared

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; disable-S `use_key`; `d0` in W; planted `when=` / `loc=` / `here=`; `w_n < 200`; empty-S green `wait` | same |
| Fail | Untrained already `use_key`; green fails; exact `{door, here}` still solves; freeze-search / use-off still `use_key`; messy-is-wait still `use_key` | Untrained already `use_key`; red stays `open`; last-50 ≈ 0; split restored |
| Store-works | Untrained `open`; unmount red `use_key` from `p99.tag`; green `wait`; exact-match / search-off / use-off fail; `w_n >= 200` | Same without splitting the return |

Do not restore exact match, plant `when=`, shrink W, or retune heads to rescue a plot.

## Headline

| Check | A split | B shared return |
|-------|---------|-----------------|
| Classification | **Store-works** | **Store-works** |
| W size | 256 | 256 |
| Untrained | `open` (clutter with code) | `open` |
| Trained red, unmount W | **`use_key`** (`p99.tag` `where=0` `pad=7`) | **`use_key`** |
| Held-out green, unmount W | **`wait`** (`p98.tag`) | **`wait`** |
| Exact-match / search-off / use-off | `open` / `open` / `open` | `open` / `open` / `open` |
| Messy-is-wait swap | `wait` | n/a (A control) |
| Train last 50 | 0.98 | 0.96 |
| Cortex | unchanged | unchanged |

## Compare

**A** is the jump: the same `{has_code, has_rare}` recipe still finds the needle in 256 unread pages. Untrained takes early clutter that carries the door integer. Trained prefers a rare key that also carries the code. Exact `{door, here}` finds nothing. Use-off commits `p99.tag` and stays `open`. Swap (same messy shape, `action=0`) stays `wait`.

**B** shared return also **Store-works** (last-50 0.96). Same caveat as TM.0.1.x: two-head shared return can work; it does not overturn v16/v23 starvation on larger joints.

## Audit (not retuned)

- Clutter `action` is OPEN only. WAIT equals door-red code `0`; using WAIT on clutter would make rare distractors falsely `has_code`. That is a W design fix, not a new search feature.
- Rare distractor keys (`extra`, `misc`, …) sit only on `place=1` files so they never share `has_code` with red or green.
- Field checks use tag boundaries (`\\nname=` / start); `"here="` is not a substring match on `where=`.
- Cortex SHA256 `57754948a40738e1afce6f5df5c4e3db4bc0462b58ba942b4f948b6fca13aec7` on both arms.
- `use_match_head` / `use_qname_head` are false. Agent module has no `"where"` literal.
- disable-S and empty S stay `open`. Empty-S green is `open`.
- B `trained_split` is false. `w_n=256` (≥ 200).

## Honest limits

- 256 `.tag` files is wiki-shaped **count**, not Wikipedia and not English.
- `{has_code, has_rare}` is still the TM.0.1.2 genome heuristic. No embeddings.
- Copy is still frozen `action=`. Combined open names + search is not claimed.
- Four acts. `where` / `pad` are made-up tags. Documents and a free life stay later.

## Reproduce

```bash
python tests/test_tm020.py
python tests/test_tm012.py
python -m experiments.run_tm020
```
