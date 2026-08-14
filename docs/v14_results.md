# v14 results: pick-one vs write schema

**Date:** 14 August 2026  
**Classification:** **A Store-works** · **B Store-works**  
**Run:** `runs/2026-08-14_093020_v14`

## Question

v13 copies `action=` from a file. Two holes remain:

- **A.** Select still **sums every match** for that door. Overwrite (`d0.tag`) hid same-door conflict.
- **B.** Write still **always** emits `{door, action}`. v9 learned WHEN, not WHAT.

Same frozen cortex. Generic copy. `n_forced=0`. Probe greedy. W has no `d0`/`d1`/`d2`. No door id in either head.

## Predeclared

### A — unique writes + pick one vs all

When one: frozen **newest `when=`**. Train pick on red (stale `wait` + later authored `use_key`). Held-out: planted green stale `open` + newer `wait`.

| ID | If |
|----|----|
| Confound | Cortex moves; disable-S still `use_key`; answer in W; door/`action=` in pick features; probe explores |
| Fail | Pick head unchanged; untrained already `use_key`; red works, green doesn’t; apply-all still `use_key` |
| Store-works | Pick changed; untrained mix `wait`; trained red `use_key`; held-out green `wait`; apply-all still mixes; empty S / disable-S `open` |

### B — schema `{door}` vs `{door, action}`

Integer still comes from the opening event. Train schema on red only. Held-out green life.

| ID | If |
|----|----|
| Confound | Cortex moves; disable-S still `use_key`; answer in W; door in schema features; probe explores |
| Fail | Schema unchanged; untrained already includes `action=` / `use_key`; red works, green doesn’t; door-only plant `use_key` |
| Store-works | Schema changed; untrained door-only `open`; trained red `use_key`; held-out green `wait`; incomplete plant `open`; empty S / disable-S `open` |

Do not put the motor act in either head. Do not restore the USE_KEY/WAIT table.

## Headline

| Check | A pick-one | B schema |
|-------|------------|----------|
| Classification | **Store-works** | **Store-works** |
| Untrained red | `wait` (stale+new summed) | `open` (note is `door=0` only) |
| Trained red | **`use_key`** (newest `d0_t8_1.tag`) | **`use_key`** (`action=2`) |
| Held-out green | **`wait`** | **`wait`** (`action=0`) |
| Control | apply-all still `wait` | door-only plant `open` |
| Empty S / disable-S | `open` | `open` |
| Head changed | pick yes | schema yes |
| Cortex SHA256 | unchanged (same as B) | unchanged (same as A) |
| Train last 50 | 0.90 | 0.78 |

A red life authored a **second** file (`d0_stale.tag` + `d0_t8_1.tag`). B untrained file has no `action=`; trained red/green files do.

## Compare

Both heads are one-bit skills over a frozen grammar. **A** is retrieve: don’t copy every matching integer. **B** is write: do put the integer on the page. Green transfer in both means the head did not memorize `red → use_key`.

A trained more cleanly (0.90 vs 0.78). B has to open the door *and* emit a complete note before the probe can succeed.

A’s green probe used **planted** conflict (stale `open` + newer `wait`), not a second free life. That isolates pick. B’s green was a free life that authored `d2.tag`.

## Honest limits

- Newest-wins and the two write templates are still genome.
- A and B clamp WHEN-to-write and the use-gate **on** so each arm tests one head.
- Unique filenames are an opt-in flag. v7–v13 still overwrite `d0.tag`.
- Not a ranker among many, not open-ended authoring, not a general tool-user.

## Reproduce

```bash
python tests/test_v14.py
python tests/test_v13.py
python -m experiments.run_v14
```
