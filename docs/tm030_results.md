# TM.0.3.0 results: a life / shared return

**Date:** 14 August 2026  
**Classification:** **A Store-works** · **B Fail**  
**Run:** `runs/2026-08-14_112946_tm030`

Not the scripted probe→unmount→probe train of TM.0.1.2 / TM.0.2.0. The agent lives: wander, search unread W, commit, act. After ρ reset with W gone, a greedy probe must still use S. Messy page `{where:0, action:2, pad:7}`. Copy frozen `action=`. Cortex frozen. `write_from_events=False`. `n_forced=0`. Agent source has no `where` string.

## Question

Can find/commit/use happen in a **free life**, then survive ρ reset without W — without restoring the experimenter unmount curriculum?

- Search `{has_code, has_rare}` (unchanged). New flag only: `record_search_on_explore` so search leaves traces while the motor explores (use traces still probe-only).
- **A.** Split: found messy `action=2` in S during life / probe after unmount+ρ reset.
- **B.** One shared return: `r` = probe correct, applied to search and use.

Held-out green: free `experience_green` life on W with `p98.tag` `{where:2, action:0, pad:7}`, then greedy `probe_green`.

## Predeclared

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; disable-S `use_key`; `d0` in W; planted `when=` / `loc=` / `here=`; writes from life; forced curriculum; empty-S green `wait` | same |
| Fail | Untrained probe already `use_key`; green fails; exact `{door, here}` still solves; freeze-search / use-off still `use_key`; messy-is-wait still `use_key`; training not a free life | Untrained already `use_key`; red stays `open`; last-50 ≈ 0; split restored |
| Store-works | Untrained probe `open`; free red life commits `p99.tag`; after ρ reset W gone → `use_key`; green `wait`; controls fail; `n_forced=0` | Same without splitting the return |

Do not restore the scripted unmount train, exact match, or plant `when=` to rescue a plot.

## Headline

| Check | A split | B shared return |
|-------|---------|-----------------|
| Classification | **Store-works** | **Fail** |
| Untrained probe after life | `open` | `open` |
| Free red life found messy page | **yes** (`p99.tag`) | no (`p0.tag`) |
| Trained red after ρ reset, W gone | **`use_key`** | `open` |
| Held-out green after reset | **`wait`** | `open` |
| Exact-match / search-off / use-off | `open` / `wait` / `open` | `open` / `open` / `open` |
| n_forced red/green | 0 / 0 | 0 / 0 |
| Train last 50 | 0.82 | **0.00** |
| Cortex | unchanged | unchanged |

## Compare

**A** is the jump: training is a free life (`experience_teach`, ε-greedy motor), not probe→unmount→probe. Search commits during life; use is scored on the post-reset probe. Red life sequence was not a forced OPEN→PICK→USE script (opens after many opens/waits, then pick, then use). Green free life opens on WAIT and leaves `p98.tag`.

**B** shared return **Fail**: last-50 0.00; red life stayed on clutter `p0.tag`; probe `open`. Heads still moved (noise updates), but credit did not compose. Same starvation pattern as v16/v23 on a harder procedure. Do not restore split credit to rescue B.

## Audit (not retuned)

- `record_search_on_explore` is required machinery: without it, `explore=True` life never records search traces (legacy “write traces only”). Default remains false for older experiments.
- Untrained free life can open the door by random affordance (`use_key` once) while committing junk `p0.tag`. Store-works is the **greedy probe after ρ reset**, which stays `open`.
- Cortex SHA256 `57754948a40738e1afce6f5df5c4e3db4bc0462b58ba942b4f948b6fca13aec7`. `write_from_events=False`. No `d0` / `loc=` / `here=` / `when=`.
- B `trained_split` is false. A `trained_life` is true.

## Honest limits

- Still four acts, messy `.tag` W (~12 files), frozen `action=` copy, `{has_code, has_rare}`.
- Free life is a short key/door episode, not an open world.
- Shared return still fails when find and use must compose without sliced credit.
- Documents / English still later.

## Reproduce

```bash
python tests/test_tm030.py
python tests/test_tm012.py
python -m experiments.run_tm030
```
