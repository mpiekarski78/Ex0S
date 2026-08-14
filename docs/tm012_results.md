# TM.0.1.2 results: messy retrieve / shared return

**Date:** 14 August 2026  
**Classification:** **A Store-works** · **B Store-works**  
**Run:** `runs/2026-08-14_111058_tm012`

No exact `loc=` / `door=` / `here=` query. Rank unread files. The useful page puts the door integer on an unknown key (`where=`) and adds an extra field (`pad=`). Exact match misses. Copy stays frozen `action=`. The agent source has no `where` string.

## Question

TM.0.1.0–0.1.1 still retrieved by exact tag match. TM.0.1.2 asks whether a boxed head can pick a file when the query is **not** `key=door_code`.

Features per file: `{has_code, has_rare}` — any integer on the file equals the current door code; any key is uncommon in the pool. No name id, no motor integer. Untrained prior: any file that carries the code (first on ties → clutter `p0.tag`). Cortex frozen. Generic copy. No `d0.tag`. No `when=`. No writes from life. Probe greedy.

- **A.** Split: found `action=2` in S / unmount probe.
- **B.** One shared return: `r` = unmount probe correct, applied to search and use.

Held-out green: `{where:2, action:0, pad:7}`. Common `place=2` clutter also carries the code; rare keys must win.

## Predeclared

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; disable-S `use_key`; `d0` in W; planted `when=` / `loc=` / `here=`; empty-S green `wait` | same |
| Fail | Untrained already `use_key`; green fails; exact `{door, here}` still solves; freeze-search / use-off still `use_key`; messy-is-wait still `use_key` | Untrained already `use_key`; red stays `open`; last-50 ≈ 0; split restored |
| Store-works | Untrained `open` (first code file); unmount red `use_key` from `p99.tag`; green `wait`; exact-match / search-off / use-off fail | Same without splitting the return |

Do not restore exact `loc=` / `door=` match or plant `when=` to rescue a plot.

## Headline

| Check | A split | B shared return |
|-------|---------|-----------------|
| Classification | **Store-works** | **Store-works** |
| Untrained | `open` (`p0.tag`) | `open` (`p0.tag`) |
| Trained red, unmount W | **`use_key`** (`p99.tag` `where=0` `pad=7`) | **`use_key`** |
| Held-out green, unmount W | **`wait`** (`p98.tag`) | **`wait`** |
| Exact-match / search-off / use-off | `open` / `wait` / `open` | `open` / `wait` / `open` |
| Messy-is-wait swap | `wait` | n/a (A control) |
| Train last 50 | 0.96 | 0.82 |
| Cortex | unchanged | unchanged |

## Compare

**A** is the jump: retrieve is a rank over files, not `world.match({loc:0})`. Untrained takes the first file that carries the door integer (clutter). Trained prefers a rare key that also carries the code. Exact `{door, here}` finds nothing. Use-off commits `p99.tag` and stays `open`. Swap (same messy shape, `action=0`) stays `wait`.

**B** shared return worked on this two-head stack (last-50 0.82). Same caveat as TM.0.1.0/0.1.1: it does not overturn v16/v23 starvation on larger joints.

## Audit (not retuned)

- First classifier pass labeled Confound because `"here=" in "where=0"`. Field checks now require a tag boundary (`\\nname=` or start of text). Retest after that fix; no weights or W were changed to rescue a plot.
- Cortex SHA256 `57754948a40738e1afce6f5df5c4e3db4bc0462b58ba942b4f948b6fca13aec7` on both arms.
- `use_match_head` / `use_qname_head` are false. Agent module has no `"where"` literal.
- disable-S and empty S stay `open`. Empty-S green is `open`.
- B `trained_split` is false.

## Honest limits

- `has_code` is still an exact integer on *some* field, not string search or similarity.
- `has_rare` is the same uncommon-key heuristic as TM.0.1.0.
- Copy is still frozen `action=`. Combined open names + search is not claimed.
- Four acts. W is a handful of `.tag` files. `where` / `pad` are made-up tags, not English.

## Reproduce

```bash
python tests/test_tm012.py
python tests/test_tm011.py
python -m experiments.run_tm012
```
