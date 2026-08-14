# TM.0.3.2 results: prose retrieve / shared return

**Date:** 14 August 2026  
**Classification:** **A Store-works** · **B Fail**  
**Run:** `runs/2026-08-14_115259_tm032`

Same free-life procedure as TM.0.3.0/0.3.1. W is **pure prose** `.md` — no filed `where=` / `action=` / `loc=` / `door=` / `here=` lines. Useful page: “At place 0 the working motor was 2.” Digits are scanned into anonymous `n0`, `n1`, … tags. A boxed vname head picks which int to copy. Cortex frozen. No NOTE-copy. `n_forced=0`.

## Question

Can find/commit/use work when the useful unread page has **no motor/place field names** — only prose with digits — then survive ρ reset with W gone?

- Genome digit scan (not English NLP). Heading `# p99` digits are **not** world content.
- Search `{has_code, has_rare}` with rare **words** in prose when keys are uniform.
- Vname features on prose ints: `{is_code, val_common}`. Untrained prefers the door code; trained must copy the other motor int (green trap: code 2 ≠ WAIT).
- **A.** Split: found door+motor ints in S during life / probe after reset.
- **B.** One shared return.

## Predeclared

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; disable-S `use_key`; filed `where=`/`action=`/… in W; `.tag` W; `d0` in W; writes from life; forced curriculum | same |
| Fail | Untrained probe already `use_key`; green fails; exact match still solves; freeze search/vname/use still `use_key`; training not a free life | Untrained already `use_key`; red stays `open`; last-50 ≈ 0; split restored |
| Store-works | Untrained probe `open`; free red commits prose ints; after ρ reset W gone → `use_key`; green `wait`; S has `n*` not `action=` | Same without splitting the return |

Do not restore filed `action=` / `where=`, NOTE-copy, or the scripted unmount train to rescue a plot.

## Headline

| Check | A split | B shared return |
|-------|---------|-----------------|
| Classification | **Store-works** | **Fail** |
| Filed where=/action= in W | no | no |
| Untrained probe after life | `open` | `open` |
| Free red found door+motor ints | **yes** (`n0=0`, `n1=2`) | no (`c00` clutter) |
| Trained red after ρ reset, W gone | **`use_key`** | `open` |
| Held-out green after reset | **`wait`** (`n0=2`, `n1=0`) | `open` |
| Exact-match / search-off / use-off | `open` / `open` / `open` | `open` / `open` / `open` |
| Train last 50 | 0.56 | **0.00** |
| Cortex | unchanged | unchanged |

## Compare

**A** is the jump: no filed motor/place keys in W. Free life ranks prose pages, commits anonymous ints, learns which int is not the door code, then after ρ reset with W gone still `use_key`. Green copies `0` not place-code `2`.

**B** shared return **Fail** (last-50 0; clutter). Same hole as TM.0.3.0/0.3.1.

## Audit (not retuned)

- First draft extracted digits from `# p10` headings, so clutter falsely carried code `0`. Fixed: ints come from body only; clutter filenames are `c00.md`… without body filename digits.
- `ProseLibrary` rejects pages that still plant filed `where=`/`action=`/… lines.
- Cortex SHA256 unchanged. S shows `n0`/`n1`, not `action=`/`where=`.

## Honest limits

- Digit scan + anonymous `n*` is still genome machinery, not reading English.
- Words like “place” / “motor” are in the notes for humans; the policy does not parse those words as field names.
- Shared return still fails. Four acts. Not Wikipedia.

## Reproduce

```bash
python tests/test_tm032.py
python tests/test_tm031.py
python -m experiments.run_tm032
```
