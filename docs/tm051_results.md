# TM.0.5.1 results: correct a wrong commit / shared return

**Date:** 14 August 2026  
**Classification:** **A Store-works** · **B Fail**  
**Run:** `runs/2026-08-14_123056_tm051`

Recipe jump: **how to correct**. Same no-integer channel-dial prose as TM.0.5.0. Search is **frozen untrained** (first remaining file) so rare-word search cannot skip the hole. Untrained keeps junk. Trained drops clutter after a failed act (not files that already name an innate motor), blacklists that W page, retries. After ρ reset, W gone, corrected S is PRESS / held-out TUNE. Cortex frozen (`a485b26b…`). `n_forced=0`.

## Question

Can the agent recover from a wrong commit — detect fail, revise S, retry, and still use the corrected file after ρ reset?

## Predeclared

| ID | A | B |
|----|---|----|
| Confound | Search trained to skip correct; digits; digit-copy; door world; writes from life | same |
| Fail | Untrained already PRESS or revises; after reset not PRESS; C not TUNE; revise-off still PRESS; n_revised=0 | Untrained PRESS; miss; empty S solves A |
| Store-works | Untrained HOLD n_revised=0; eval revises ≥1; after reset PRESS; C TUNE; revise-off HOLD | Same without split credit |

## Headline

| Check | A split | B shared return |
|-------|---------|-----------------|
| Classification | **Store-works** | **Fail** |
| Untrained probe / n_revised | `hold` / 0 | `hold` / 0 |
| Eval life n_revised | **11** (dropped c00–c10) | 0 (`c00` kept) |
| S after life | `p99` `w2=press` | `c00` clutter |
| After ρ reset, W gone | **`press`** | `hold` |
| Held-out C | **`tune`** (`p98`, n_revised 11) | `hold` |
| Revise-off / use-off / swap | `hold` / `hold` / `idle` | `hold` / `hold` / `hold` |
| Search head | unchanged | unchanged |
| Train last 50 | 0.60 | **0.00** |
| Cortex | unchanged | unchanged |

## Compare

**A** is the jump: first commit is clutter; the world says no; junk is dropped; the useful page is kept; after ρ reset the corrected S steers. Revise-off stays on clutter — search/copy alone did not rescue.

**B** shared return **Fail** (last-50 0; never revises; still `c00`). Same credit hole.

## Audit (not retuned)

- TM.0.5.0 still **Store-works** / **Fail** without revise.
- Search frozen: `search_changed=False`. Walk is filename order, not rare-word skip.
- Experience life does not abort on lucky PRESS (probe still one-shot).
- Genome will not delete a file that already names an innate act (`press` / `tune` / `idle`).
- Door default and digit-copy off.

## Honest limits

- Correction is drop+skip, not overwrite or `ok=` mark.
- Keep-if-act-name is body vocabulary again, not English.
- Search frozen is an experimenter clamp so the jump is really correct, not 0.5.0 again.
- Shared return still fails. Not Wikipedia.

## Reproduce

```bash
python tests/test_tm051.py
python tests/test_tm050.py
python -m experiments.run_tm051
```
