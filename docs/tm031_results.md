# TM.0.3.1 results: documents / shared return

**Date:** 14 August 2026  
**Classification:** **A Store-works** · **B Fail**  
**Run:** `runs/2026-08-14_114403_tm031`

Same free-life procedure as TM.0.3.0. W is **`.md` documents** with prose plus embedded `k=v` lines — not tidy `.tag` files. Useful page `p99.md` has prose and `{where:0, action:2, pad:7}`. Exact `loc=` / `door=` / `here=` still miss. Copy frozen `action=`. Cortex frozen. `write_from_events=False`. `n_forced=0`. No NOTE-copy prior. Agent source has no `where` string.

## Question

Can find/commit/use in a free life work when unread W is markdown documents, then survive ρ reset with W gone?

- `DocLibrary` loads `*.md` only. Integer fields still come from `k=v` lines in the file (prose lines without `=` are ignored). Not English understanding; not LSTM-in-prompt.
- Search `{has_code, has_rare}` unchanged. `record_search_on_explore` as in TM.0.3.0.
- **A.** Split: found messy `action=2` in S during life / probe after unmount+ρ reset.
- **B.** One shared return: `r` = probe correct.

Held-out green: free life on W with `p98.md` `{where:2, action:0, pad:7}`.

## Predeclared

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; disable-S `use_key`; `.tag` files in W; no prose; `d0` in W; planted `when=` / `loc=` / `here=`; writes from life; forced curriculum | same |
| Fail | Untrained probe already `use_key`; green fails; exact `{door, here}` still solves; freeze-search / use-off still `use_key`; training not a free life | Untrained already `use_key`; red stays `open`; last-50 ≈ 0; split restored |
| Store-works | Untrained probe `open`; free red commits from `p99.md`; after ρ reset W gone → `use_key`; green `wait`; controls fail; W all `.md` | Same without splitting the return |

Do not restore `.tag` W, NOTE-copy, exact match, or the scripted unmount train to rescue a plot.

## Headline

| Check | A split | B shared return |
|-------|---------|-----------------|
| Classification | **Store-works** | **Fail** |
| W | all `.md`, has prose, no `.tag` | same |
| Untrained probe after life | `open` | `open` |
| Free red found messy doc | **yes** (`p99` → S) | no (`p0`) |
| Trained red after ρ reset, W gone | **`use_key`** | `open` |
| Held-out green after reset | **`wait`** | `open` |
| Exact-match / search-off / use-off | `open` / `wait` / `open` | `open` / `open` / `open` |
| Train last 50 | 0.82 | **0.00** |
| Cortex | unchanged | unchanged |

## Compare

**A** is the jump: unread W is documents. Free life ranks `.md` pages, commits the useful scrap into S (as `.tag` after commit — inspectable store form), then after ρ reset with W gone the probe still `use_key`. Exact match finds nothing in the docs.

**B** shared return **Fail** again (last-50 0; clutter `p0`). Same hole as TM.0.3.0 / v16 / v23. Do not restore split credit to rescue B.

## Audit (not retuned)

- `DocLibrary` is new; `TagLibrary` unchanged for older runs.
- Integers still come from `k=v` lines inside `.md`, not from reading English prose. Honest limit: documents are a **container**, not NLP.
- No NOTE-copy. No LanguageAgent path. Cortex SHA256 `57754948a40738e1afce6f5df5c4e3db4bc0462b58ba942b4f948b6fca13aec7`.
- B `trained_split` is false. A `trained_life` is true. `w_has_tag_files` is false.

## Honest limits

- Still four acts, `{has_code, has_rare}`, frozen `action=` copy.
- `.md` with embedded tags is not Wikipedia and not reading English.
- Shared return still fails on free life.
- Pure prose without `k=v` lines is not claimed.

## Reproduce

```bash
python tests/test_tm031.py
python tests/test_tm030.py
python -m experiments.run_tm031
```
