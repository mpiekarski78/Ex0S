# v17 results: read `do=` vs match `here=`

**Date:** 14 August 2026  
**Classification:** **A Store-works** · **B Store-works**  
**Run:** `runs/2026-08-14_094824_v17`

## Question

v13–v16 still **named** the slots: copy `action=`, match `door=`.

- **A.** Write `do=` (same integer). Learn which key to copy: `{action, do}`. Untrained reads `action=` (missing → `open`).
- **B.** Write `here=` (same place code). Learn which key to match: `{door, here}`. Untrained matches `door=` (no hit → `open`).

Train on red. Held-out green. Split credit. Cortex frozen. W has no answers. `n_forced=0`. Probe greedy. No door id or motor integer in either head. Do not start reading `action=` / matching `door=` to rescue a plot.

## Predeclared

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; disable-S `use_key`; door/integer in the key head | Cortex moves; disable-S `use_key`; door id in the match head |
| Fail | Planted `do=2` already `use_key`; green fails; `action=` control still works | Planted `here=` already `use_key`; green fails; `door=` control still works |
| Store-works | Untrained `open`; trained red `use_key` from `do=2`; green `wait`; `action=` control `open` | Untrained `open`; trained red `use_key` from `here=`; green `wait`; `door=` control `open` |

## Headline

| Check | A read `do=` | B match `here=` |
|-------|--------------|-----------------|
| Classification | **Store-works** | **Store-works** |
| Untrained (copy on, planted alt name) | `open` | `open` |
| Trained red | **`use_key`** (`do=2`, `door=0`) | **`use_key`** (`here=0`, `action=2`) |
| Held-out green | **`wait`** | **`wait`** |
| Control (other heads trained, this head untrained) | `action=` → `open` | `door=` → `open` |
| Train last 50 | 0.74 | 0.76 |
| Cortex | unchanged | unchanged |

## Compare

Both are one-bit **name** skills over a frozen two-key menu. **A** is “which integer field to copy.” **B** is “which place field to match.” Green transfer means the head did not memorize `red → use_key`. The control (untrained name, trained write/use) shows the new name is required.

## Honest limits

- The menus `{action, do}` and `{door, here}` are still genome. Write is frozen to emit the alt name; the head only learns to look there.
- Not open-ended schema. Not a new motor act. Not a matcher over arbitrary tags.
- Untrained readout uses `force_use` so a missing name shows as `open` rather than a silent use-gate.

## Reproduce

```bash
python tests/test_v17.py
python tests/test_v16.py
python -m experiments.run_v17
```
