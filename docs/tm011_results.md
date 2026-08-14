# TM.0.1.1 results: open copy names / shared return

**Date:** 14 August 2026  
**Classification:** **A Store-works** · **B Store-works**  
**Run:** `runs/2026-08-14_110506_tm011`

Query is frozen to the files’ place key (`loc=` in the experiment, not a `{door, here}` menu). Copy names come from keys on the hit. The agent source has no `act` string and does not run the `{action, do}` copy menu.

## Question

TM.0.1.0 opened query names. Copy was still frozen `action=`. TM.0.1.1 opens the other name and does **not** stack the query-name head.

Candidates are the tag keys on the matched file (minus bookkeeping). Features per candidate: `{is_query, key_common}` — no name id, no integer. Untrained prior copies the query key (the place code). Cortex frozen. Generic copy. Exact match. No `d0.tag`. No `do=`. No `when=`. No writes from life. Probe greedy.

W: clutter `{place, action}` plus useful `{loc:0, act:2}` as `p99.tag`. `act` is an experiment-file name, not an agent constant.

Green is the transfer trap: `loc=2` is the same integer as USE_KEY. Copying the place field on green would `use_key`. The motor field is `act=0` (`wait`).

- **A.** Split: chosen copy-key’s value is 2 / unmount probe.
- **B.** One shared return: `r` = unmount probe correct, applied to vname and use.

## Predeclared

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; disable-S `use_key`; `d0` in W; planted `when=` / `do=`; empty-S green `wait` | same |
| Fail | Untrained already `use_key`; green fails (copied place code 2); `{action, do}` menu restored and still solves; freeze-vname / use-off still `use_key`; act-is-wait still `use_key` | Untrained already `use_key`; red stays `open`; last-50 ≈ 0; split restored |
| Store-works | Untrained `open` (copies query key); unmount red `use_key` from `act=2`; green `wait`; menu / vname-off / use-off fail | Same without splitting the return |

Do not restore `{action, do}` or plant `do=` / `action=` as the motor field to rescue a plot.

## Headline

| Check | A split | B shared return |
|-------|---------|-----------------|
| Classification | **Store-works** | **Store-works** |
| Untrained | `open` (copy `loc`) | `open` (copy `loc`) |
| Trained red, unmount W | **`use_key`** (`p99.tag` `act=2`) | **`use_key`** (`act=2`) |
| Held-out green, unmount W | **`wait`** (`act=0`, not `loc=2`) | **`wait`** |
| `{action, do}` menu / vname-off / use-off | `open` / `wait` / `open` | `open` / `wait` / `open` |
| act-is-wait swap | `wait` | n/a (A control) |
| Train last 50 | 0.90 | 0.90 |
| Cortex | unchanged | unchanged |

## Compare

**A** is the jump: the copy name is chosen among keys on the hit. Untrained copies the query key (`loc=0` → wait bias if use were on; use off stays `open`). Trained copies the other integer field. Green would `use_key` if they copied `loc=2`; they copy `act=0` and `wait`. Restored `{action, do}` menu looks for `action=` and misses. Use-off commits `p99.tag` and stays `open`.

**B** shared return worked again on a two-head stack (last-50 0.90). Same caveat as TM.0.1.0: this does not overturn v16/v23 starvation on larger joints.

## Audit (not retuned)

- Cortex SHA256 `57754948a40738e1afce6f5df5c4e3db4bc0462b58ba942b4f948b6fca13aec7` on both arms.
- `use_key_head` is false. `use_qname_head` is false. Query is `place_key="loc"` from the experiment file, not from agent source.
- Agent module has no `"act"` / `'act'` literal.
- disable-S and empty S stay `open`. Empty-S green is `open`.
- B `trained_split` is false.

## Honest limits

- Query was frozen to one file-key for this run. Combined open query+copy is not claimed.
- `{is_query, key_common}` is still a genome heuristic (prefer not to copy the index you matched on).
- Exact match. Four acts. `act` is a made-up tag, not English. W is a handful of `.tag` files.

## Reproduce

```bash
python tests/test_tm011.py
python tests/test_tm010.py
python -m experiments.run_tm011
```
