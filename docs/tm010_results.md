# TM.0.1.0 results: open query names / shared return

**Date:** 14 August 2026  
**Classification:** **A Store-works** · **B Store-works**  
**Run:** `runs/2026-08-14_105832_tm010`

First post-toy series. Not v24. Query field names come from keys that exist in W/S files. The agent source has no `loc` string and does not run the `{door, here}` match menu.

## Question

v17–v23 matched `door=` or `here=` (a genome pair). TM.0.1.0 drops that menu.

Candidates are the tag keys in the unread files (minus bookkeeping `source` / `when` / `ok`). Features per candidate: `{has_hit, key_common}` — no name id, no door id, no integer. Untrained prior prefers any hit (ties → first sorted key). Copy is still generic `action=` (value names not opened). Cortex frozen. Exact match. No `d0.tag`. No `here=`. No `when=`. No writes from life. Probe greedy.

W: clutter `{place, action}` plus useful `{loc:0, action:2}` as `p99.tag`. `loc` is an experiment-file name, not an agent constant.

- **A.** Split: found `action=2` in S / unmount probe.
- **B.** One shared return: `r` = unmount probe correct, applied to qname and use.

Held-out green: `{loc:2, action:0}`.

## Predeclared

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; disable-S `use_key`; `d0` in W; planted `when=` / `here=`; empty-S green `wait` | same |
| Fail | Untrained already `use_key`; green fails; `{door, here}` menu restored and still solves; freeze-qname / use-off still `use_key`; loc-is-wait still `use_key` | Untrained already `use_key`; red stays `open`; last-50 ≈ 0; split restored |
| Store-works | Untrained `open` (queries a file key, not `loc`); unmount red `use_key` from `p99.tag` `{loc, action}`; green `wait`; menu / qname-off / use-off fail | Same without splitting the return |

Do not restore `{door, here}` or plant `here=` / `when=` to rescue a plot.

## Headline

| Check | A split | B shared return |
|-------|---------|-----------------|
| Classification | **Store-works** | **Store-works** |
| Untrained | `open` (query `action`, `p0.tag`) | `open` (query `action`) |
| Trained red, unmount W | **`use_key`** (`p99.tag` `loc=0`) | **`use_key`** (`p99.tag` `loc=0`) |
| Held-out green, unmount W | **`wait`** (`p98.tag` `loc=2`) | **`wait`** (`p98.tag`) |
| `{door, here}` menu / qname-off / use-off | `open` / `wait` / `open` | `open` / `wait` / `open` |
| loc-is-wait swap | `wait` | n/a (A control) |
| Train last 50 | 0.88 | 0.90 |
| Cortex | unchanged | unchanged |

## Compare

**A** is the jump: the query name is chosen among keys the files actually have. Untrained hits the common key `action=` and commits clutter. Trained prefers an uncommon key that hits — in this W that is `loc`. The restored `{door, here}` menu finds nothing (`store_len=0`). Use-off commits `p99.tag` and stays `open`. loc-is-wait stays `wait`.

**B** is the training claim on this two-head stack. Shared return **worked** (last-50 0.90). That does not overturn v16 B or v23 B: those were three/four-head joints that starved. Here, once S holds the useful page, the only hitting key is usually the right index, so retrieve-after-commit is easy and one `r` can move both heads.

## Audit (not retuned)

- Cortex SHA256 `57754948a40738e1afce6f5df5c4e3db4bc0462b58ba942b4f948b6fca13aec7` on both arms.
- `use_match_head` is false. `place_key` remains `"door"` but is unused on the open-name path.
- Agent module has no `"loc"` / `'loc'` literal. Candidates at untrained red: `action`, `loc`, `place`.
- disable-S and empty S stay `open`. Empty-S green is `open`.
- B `trained_split` is false. Do not treat B as a silent split.

## Honest limits

- Value copy is still frozen `action=`. The other menu `{action, do}` was not opened.
- `{has_hit, key_common}` is still a genome heuristic (prefer uncommon indexes). It is not search and not an unbounded schema.
- After commit, S has two keys and the useful index is often the only hit. That makes unmount easier than the mounted collect step.
- Exact match on the door integer. Four acts. W is still a handful of `.tag` files. `loc` is a made-up tag, not English.

## Reproduce

```bash
python tests/test_tm010.py
python tests/test_v23.py
python -m experiments.run_tm010
```
