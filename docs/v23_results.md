# v23 results: joint find+complete+use / shared return

**Date:** 14 August 2026  
**Classification:** **A Store-works** · **B Fail**  
**Run:** `runs/2026-08-14_104353_v23`

## Question

v22 A kept complete-vs-stub without `when=`. v22 B composed find+newest+use under **split** credit, and still used planted recency.

v23 puts find + complete-vs-stub + use on the **same** unread W, with **no** `when=`, and asks whether split credit is load-bearing on that wiki path (v16 B analogue). Cortex frozen. Generic copy. Exact match. No `d0.tag`. No writes from life. Probe greedy.

W: stub `aaa.tag` `{here:0}` (filename-first); complete `p99.tag` `{here:0, action:2}`; `door=` junk `junk.tag` `{door:0, action:0}`. No recency field.

Same `make()` both arms: `collect_mode=commit`, `use_read`, `use_match_head`, `use_wcomp_head`, `force_use=False`, `place_key=door` (match must learn `here=`).

- **A.** Split credit: found `here=` / kept `action=2` / unmount probe.
- **B.** One shared return: `r` = unmount probe correct, applied to match, wcomp, and use.

Held-out green: stub `{here:2}`, complete `{here:2, action:0}`, `door=` junk `open`.

## Predeclared

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; disable-S `use_key`; `d0` in W; planted `when=`; empty-S green `wait` | same |
| Fail | Untrained already `use_key`; green fails; freeze-match `door=` / freeze-stub / use-off still `use_key`; stub-only or complete-is-junk `use_key` | Untrained already `use_key`; red stays `open`; last-50 ≈ 0; split restored to rescue |
| Store-works | Untrained `open`; unmount red `use_key` from complete `here=` page; green `wait`; three freeze-offs fail | Same joint as A without splitting the return |

Do not plant `when=` or restore split credit to rescue B.

## Headline

| Check | A split joint | B shared return |
|-------|---------------|-----------------|
| Classification | **Store-works** | **Fail** |
| Untrained | `open` (`junk.tag`, `door=`) | `open` (`junk.tag`) |
| Trained red, unmount W | **`use_key`** (`p99.tag`, no `when=`) | `open` (`junk.tag`) |
| Held-out green, unmount W | **`wait`** (`p98.tag`) | `open` (`junkg.tag`) |
| Match `door=` / stub-first / use-off | `wait` / `open` / `open` | n/a (never left prior) |
| Complete-is-junk / stub-only | `wait` / `open` | n/a |
| Train last 50 | 0.84 | **0.00** |
| Policy updates | 2480 | **0** |
| Cortex | unchanged | unchanged |

## Compare

**A** is the joint: v20 find and v22 complete run together without recency and without clamping `here=` or `force_use`. Each freeze-off control fails, so all three heads are load-bearing. Swap (complete file is `wait`) stays `wait`; stub-only stays `open`.

**B** is the training claim: that joint still needs split credit. Shared return never left last-50 0. `update()` skips zero advantage, so match/wcomp/use hashes did not move. Red stayed `door=` junk. Same starvation as v16 B, now on unread W. Do not treat A’s split as a silent clamp; B is the arm that forbids it.

## Audit (not retuned)

- Cortex SHA256 `57754948a40738e1afce6f5df5c4e3db4bc0462b58ba942b4f948b6fca13aec7` on both arms.
- W has no `when=`, no `d0`/`d1`/`d2`. disable-S and empty S stay `open`. Empty-S green is `open` (held-out is a transfer test).
- Untrained is not `use_key`. A’s untrained commit is `junk.tag` (`place_key=door`), not the stub; that is the match prior, not a leak.
- `junk.tag` has `action=`, so on a `door=` hit pool it counts as complete. Freeze-match + trained wcomp+use copies junk `wait`. That is the control, not a bug.
- Unmount probe has no W, so wcomp is not re-decided there. Training traces for wcomp come from the mounted collect step. Expected.
- B’s occasional `p99.tag` under ε did not yield `r_use=1`: the unmount probe still needs `here=` **and** use-on to read the file. Shared `r` never fired. Not rescued.

## Honest limits

- Complete-rule is still genome (first hit that has `action=` / `do=`). The head only learns *when* to use it vs filename-first.
- Two-key match menu. Exact match. Four acts. Split credit on A. Not search, not English, not Wikipedia.

## Reproduce

```bash
python tests/test_v23.py
python tests/test_v22.py
python tests/test_v20.py
python -m experiments.run_v23
```
