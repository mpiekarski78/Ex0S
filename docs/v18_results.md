# v18 results: write `do=` vs write `here=`

**Date:** 14 August 2026  
**Classification:** **A Store-works** · **B Store-works**  
**Run:** `runs/2026-08-14_095403_v18`

## Question

v17 froze the **writer** to emit `do=` / `here=` and only trained the **reader**. v18 flips that: the menus stay frozen `{action, do}` and `{door, here}`, but the boxed head learns **which name to write**. Read/match stay frozen to the alt name.

- **A.** Write-key: emit `action=` vs `do=`. Read frozen to `do=`. Untrained writes `action=` → no `do=` → `open`.
- **B.** Write-place: emit `door=` vs `here=`. Match frozen to `here=`. Untrained writes `door=` → no hit → `open`.

Train on red. Held-out green. Split credit. Cortex frozen. W has no answers. `n_forced=0`. Probe greedy. No door id or motor integer in either write-name head. Do not freeze-write `do=` / `here=` to rescue a plot.

## Predeclared

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; disable-S `use_key`; answer in W; door/integer in write-key head; probe explores | same; door id in write-place head |
| Fail | Untrained already writes `do=` / `use_key`; red works, green doesn’t; freeze-write `do=` to rescue | Untrained already writes `here=` / `use_key`; green fails; freeze-write `here=` to rescue |
| Store-works | Untrained `open` (file has `action=` not `do=`); trained red `use_key` from authored `do=2`; held-out green `wait`; empty S / disable-S `open` | Untrained `open` (file has `door=` not `here=`); trained red `use_key` from `here=`; green `wait`; empty/disable `open` |

## Headline

| Check | A write `do=` | B write `here=` |
|-------|---------------|-----------------|
| Classification | **Store-works** | **Store-works** |
| Untrained (force-write, name head off) | `open` (`action=2`, `door=0`) | `open` (`action=2`, `door=0`) |
| Planted alt name (frozen read/match) | `use_key` (`do=2`) | `use_key` (`here=0`) |
| Trained red | **`use_key`** (`do=2`, `door=0`) | **`use_key`** (`here=0`, `action=2`) |
| Held-out green | **`wait`** (`do=0`, `door=2`) | **`wait`** (`here=2`, `action=0`) |
| Control (write+use trained, name head untrained) | `action=` → `open` | `door=` → `open` |
| Train last 50 | 0.88 | 0.82 |
| Cortex | unchanged | unchanged |

## Compare

v17 was “which name to **look at**.” v18 is “which name to **emit**.” Both are one-bit name skills over a frozen two-key menu. Green transfer means the head did not memorize `red → use_key`. The control (untrained name, trained write/use) shows the new name is required on the write path. Split credit: write / use / write-name separately. Shared return is still not restored.

## Honest limits

- The menus `{action, do}` and `{door, here}` are still genome. Read is frozen to `do=`; match is frozen to `here=`. Only the writer learns which label to put on disk.
- Not open-ended schema. Not a new motor act. Not a matcher over arbitrary tags.
- Untrained write uses `force_write` so the default name is visible (`action=` / `door=`). Training itself is unclamped (`force_write` / `force_use` off).

## Reproduce

```bash
python tests/test_v18.py
python tests/test_v17.py
python tests/test_v13.py
python tests/test_v7.py
python -m experiments.run_v18
```
