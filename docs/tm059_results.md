# TM.0.5.9 results: correct dirty S / shared return

**Date:** 14 August 2026  
**Classification:** **A Store-works** · **B Fail**  
**Run:** `runs/2026-08-14_142519_tm059`

Recipe jump: **correct the dirty store.** Same 64-page multi-rare never-wipe W as TM.0.5.8. Once S names here, stop committing. After a successful stamp, drop pages that never got an act name. After train, S has **1 file** (`xenon`+`press`+`cha`), n_revised=31, probe **PRESS** / foil C **HOLD**. C life on that S: A **PRESS** kept, C **TUNE** added. Wipe-between loses A. Cortex frozen (`a485b26b…`). `n_forced=0`. `n_train` was not raised. New flags default **off**.

Skipped on purpose: dropping `has_code`, removing `domain=`, English, solving B.

## Question

Can the organism stop stamp-collecting on a never-wipe life, or does S only grow?

## Predeclared

| ID | A | B |
|----|---|----|
| Confound | Cortex moves; W still a dozen logs; English; drop `has_code` / `domain=`; revise/here-only on by default | same |
| Fail | Dirty train S not PRESS; n files > 8; n_revised=0; C life loses A or misses TUNE; wipe-between still PRESS on A | Untrained PRESS; dirty S miss; split restored |
| Store-works | After train A PRESS / C HOLD, S ≤8 files, n_revised≥1; C on that S A PRESS / C TUNE; wipe-between A HOLD | Same without split |

## Headline

| Check | A split | B shared return |
|-------|---------|-----------------|
| Classification | **Store-works** | **Fail** |
| Untrained A / foil C | `hold` / `hold` | `hold` / `hold` |
| After train, S: A / foil C | **`press` / `hold`** | `press` / `hold` |
| C life on that S: A / C | **`press` / `tune`** | `press` / `hold` |
| Wipe-between: A / C | **`hold` / `tune`** | `hold` / `hold` |
| Train S n files | **1** | **1** |
| n_revised train | 31 | 18 |
| Train last 50 | 0.92 | **0.98** |
| Cortex | unchanged | unchanged |

## Compare

**A** is the jump: TM.0.5.8’s dirty store (19 files) is not required. After a real stamp the unstamped scraps go; once S names here, W is not vacuumed. One inspectable note still PRESS. A later C life writes `tune`+`chc` beside it.

**B** shared return **Fail**: C life did not stamp `tune`. Last-50 **0.98** on the first fact. Not retuned.

## Audit (not retuned)

- TM.0.5.0–0.5.8 unit tests still pass.
- Default door agent: here-match / annotate / rare-commit / here-only / revise **off**.
- TM.0.5.8 flags unchanged (revise and here-only default off).
- Search still computes `has_code`. `domain="dial"` still set. No English lexicon (`push` absent).

## Honest limits

- Search is still binary `{has_code, has_rare}`. 64 short logs, not Wikipedia ranking.
- Sweep is frozen grammar: unstamped pages are not facts. It is not English NLP.
- Shared return still misses the second fact. Split credit stays load-bearing for two facts.
- Still five acts, innate names, `domain=`.

## Reproduce

```bash
python tests/test_tm059.py
python tests/test_tm058.py
python tests/test_tm057.py
python tests/test_tm056.py
python tests/test_tm055.py
python tests/test_tm054.py
python tests/test_tm053.py
python tests/test_tm052.py
python tests/test_tm051.py
python tests/test_tm050.py
python -m experiments.run_tm059
```
