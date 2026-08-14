# v20 results: find unread W vs find vs junk

**Date:** 14 August 2026  
**Classification:** **A Store-works** · **B Store-works**  
**Run:** `runs/2026-08-14_102653_v20`

## Question

v9–v19 turned collect off and made the agent **author** S. The long-term claim is the opposite of stuffing wiki into weights: **unread available data (W)** may be found and committed. v5/v8 already committed from W when the file used the genome name `door=`. v20 asks whether a boxed matcher can **find** a page that is not on that default index.

W pages are not `d0.tag`. Collect is the frozen v5 rule (S miss + W hit → commit). Use-gate may learn. Cortex frozen. Generic copy. Split credit (found `here=` in S / unmount probe). No writes from life. Probe greedy.

- **A.** W has one useful page `{here:0, action:2}` as `p99.tag` among clutter. Untrained matches `door=` → miss → `open`.
- **B.** Same page **plus** junk `{door:0, action:0}` as `junk.tag`. Untrained hits junk (and may commit it) but the use-gate is off → `open`. Trained must query `here=` or junk `wait` leaks after the use-gate turns on.

Held-out green is a different unread page `{here:2, action:0}` (`p98.tag`), never in train W. B’s green W also has junk `{door:2, action:1}` (`open`).

## Predeclared

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; disable-S `use_key`; answer filename `d0`/`d2` in W; empty-S green already `wait`; writes from life | same, and B W missing junk |
| Fail | Untrained already `use_key` or already `here=`; red unmount wrong; green fails; freeze-match `here=`; trained use + untrained `door=` still `use_key` | same, plus junk-only W `use_key` |
| Store-works | Untrained `open`; commit `p99.tag`; unmount W; red `use_key`; green `wait`; `door=` control `open` | Untrained `open`; red `use_key` from `here=`; green `wait`; `door=` control `wait` (junk); junk-only `open` |

Do not freeze-match `here=` or restore `d0.tag` / USE_KEY table to rescue a plot. Do not put Wikipedia in weights. This is still a handful of `.tag` files.

## Headline

| Check | A find | B find vs junk |
|-------|--------|----------------|
| Classification | **Store-works** | **Store-works** |
| Untrained | `open` (`match_alt` false; no `here=` in S) | `open` (committed `junk.tag`; use-gate off) |
| Trained red, unmount W | **`use_key`** (`p99.tag` `here=0 action=2`) | **`use_key`** (same; not junk) |
| Held-out green, unmount W | **`wait`** (`p98.tag` `here=2 action=0`) | **`wait`** (same; not `open` junk) |
| Match control (use trained, match `door=`) | `open` | `wait` (junk) |
| Junk-only W | n/a | `open` (`here=` misses junk) |
| Empty S / disable-S | `open` | `open` |
| Train last 50 | 0.84 | 0.90 |
| Cortex | unchanged | unchanged |

## Compare

A learns to query an index that is not the genome default. B learns the same, **against** a page that sits on that default and would copy `wait` once the use-gate is on. The `door=` control on B is `wait`, not `use_key`, so they did not ignore junk by baking `use_key` into the policy. Junk-only W stays `open` because `here=` misses. Green used `here=` on a page that was never in train W.

## Audit

Two agent bugs would have made find impossible even with a working trainer:

1. **Match ran after collect.** W lookup used a stale `match_alt` (or `door=`). Same-step `here=` never saw `p99.tag`.
2. **`collect` replaced `last_policy`. ** Even after reordering, the collect decision wiped `match_alt` before retrieve.

Both are fixed. v17–v19 tests still pass (they use `collect_mode=off`; retrieve already used same-step match).

A first trainer (`collect_mode=policy`, one shared probe return) never left last-50 0.00: `update()` skips zero advantage, and commit+apply+`here=` never co-occurred. That is the v16 shared-return failure, not a reason to freeze `here=`. Retest uses frozen v5 commit-on-hit and split credit (found `here=` vs unmount probe). Still **Store-works** / **Store-works**.

## Honest limits

- Collect is not re-learned (v8 already did). Peek is empty-S, not a learned peek head.
- The useful page still uses `action=` (payload grammar frozen). Only the **query name** is learned, and the menu is still `{door, here}`.
- Not Wikipedia. Not English. Not a general searcher over arbitrary tags.
- Split credit; shared probe-return was not restored.

## Reproduce

```bash
python tests/test_v20.py
python tests/test_v19.py
python tests/test_v8.py
python -m experiments.run_v20
```
