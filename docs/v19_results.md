# v19 results: shared value-name vs shared place-name

**Date:** 14 August 2026  
**Classification:** **A Store-works** · **B Store-works**  
**Run:** `runs/2026-08-14_101202_v19`

## Question

v17 froze the writer and trained the reader. v18 froze the reader and trained the writer. Untrained stayed `open` only because the two sides **disagreed**.

v19 freezes neither side. Menus stay `{action, do}` and `{door, here}`. Untrained priors are mismatched (write `action=` / `door=`, read/match `do=` / `here=`). They must learn a **shared name**. Either equilibrium is legal.

- **A.** Write-key and read-key both learn. Untrained: file has `action=`, reader looks at `do=` → `open`.
- **B.** Write-place and match-key both learn. Untrained: file has `door=`, matcher looks at `here=` → `open`.

Train on red. Held-out green. Split credit (write / use / agree). Cortex frozen. W has no answers. `n_forced=0`. Probe greedy. No door id or motor integer in the heads. Do not freeze one side to rescue a plot.

## Predeclared

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; disable-S `use_key`; answer in W; door/integer in a name head; empty-S green already `wait` | same |
| Fail | Untrained already `use_key`; red works, green doesn’t; write+use without agreement still works; freeze a head off | same |
| Store-works | Untrained `open`; trained red `use_key` from one shared value name; green uses that same name and `wait`; mismatch control `open` | Untrained `open`; trained red `use_key` from one shared place name; green same name and `wait`; mismatch control `open` |

## Headline

| Check | A shared value-name | B shared place-name |
|-------|---------------------|---------------------|
| Classification | **Store-works** | **Store-works** |
| Untrained (mismatched priors) | `open` (`action=2`, reader on `do=`) | `open` (`door=0`, matcher on `here=`) |
| Planted alt / old name | `do=` → `use_key`; `action=` → `open` | `here=` → `use_key`; `door=` → `open` |
| Trained red | **`use_key`** (`do=2`, `door=0`) | **`use_key`** (`action=2`, `door=0`) |
| Agreed name | **`do=`** (writer moved to the reader) | **`door=`** (matcher moved to the writer) |
| Held-out green | **`wait`** (`do=0`) | **`wait`** (`action=0`, `door=2`) |
| Control (write+use trained, names untrained) | `open` | `open` |
| Empty-S green | `open` | `open` |
| Train last 50 | 0.74 | 0.82 |
| Cortex | unchanged | unchanged |

## Compare

They did **not** pick the same side of the menu. A met on the reader’s prior (`do=`). B met on the writer’s prior (`door=`). Both are legal conventions. Green used the same name as red, so the heads did not memorize `red → use_key`. Write+use without agreement still `open`, so the shared name is required.

## Honest limits

- The menus `{action, do}` and `{door, here}` are still genome. This is a one-bit convention, not open-ended schema.
- Untrained mismatch is a designed prior (`b_key` / `b_match` = +1.2). Default `UsePolicy` still prefers the old names (v17/v18 unchanged).
- Not a new motor act. Not a matcher over arbitrary tags. Split credit; shared return is still not restored.

## Reproduce

```bash
python tests/test_v19.py
python tests/test_v18.py
python tests/test_v17.py
python -m experiments.run_v19
```
