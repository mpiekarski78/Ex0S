# TM.0.6.2 results: never-wipe English / shared return

**Date:** 14 August 2026  
**Classification:** **A Fail** · **B Store-works**  
**Run:** `runs/2026-08-14_175257_tm062`

Recipe jump: **never-wipe English life** on the TM.0.6.1 one-bind recipe. Same tiny English W (`push`+`argon`, `adjust`+`alpha`). Train does not rmtree S. After 500 A lives the dirty store (no fresh A life) probes **PRESS** / foil C **HOLD** from `bind=push`. A later C life on that S committed `p98` (`adjust`/`alpha`) but **did not stamp** `did=tune` / `bind=adjust`; probe C stayed **HOLD**. Wipe-between loses A. Cortex frozen (`a485b26b…`). `n_forced=0`. `n_train` was not raised. `use_commit_rare_only` on this slice only (default off). Revise / here-only stayed off.

Skipped on purpose: dropping `has_code`, removing `domain=`, math, solving B (B happened to pass; not retuned), turning on 0.5.9 correct flags.

## Question

Can the English one-bind fact written during training survive 500 more episodes in the same store, and can a C life add a second bind on that dirty S — or does English still need the experimenter to wipe?

## Predeclared

| ID | A | B |
|----|---|----|
| Confound | Cortex moves; unique-rare needle; W names an innate motor; `push` in the agent; drop `has_code` / `domain=`; revise/here-only smuggled on; train still wipes | same |
| Fail | Untrained PRESS; train S not PRESS; C life loses A or misses TUNE; wipe-between still PRESS on A; bind=`argon`; nonce PRESS | Untrained PRESS; A miss; C miss; nonce PRESS |
| Store-works | Untrained HOLD; never-wipe train A PRESS / C HOLD from `push`; C life on that S A PRESS / C TUNE; wipe-between A HOLD; nonce HOLD; bind-all nonce PRESS | Same without splitting the return |

## Headline

| Check | A split | B shared return |
|-------|---------|-----------------|
| Classification | **Fail** | **Store-works** |
| Untrained A / foil C | `hold` / `hold` | `hold` / `hold` |
| After train, dirty S: A / foil C | **`press` / `hold`** | **`press` / `hold`** |
| C life on dirty S: A / C | **`press` / `hold`** | **`press` / `tune`** |
| Wipe-between A | **`hold`** | **`hold`** |
| Bind-off A | **`hold`** | **`hold`** |
| Nonce-only A | **`hold`** | **`hold`** |
| Bind-all nonce A | **`press`** | **`press`** |
| Train last 50 | 0.92 | 0.92 |
| Cortex | unchanged | unchanged |

Inspectable train note: `bind=push`, `did=press`, `w*=argon|push|cha`. After A's C life, `p98.tag` is in S with `adjust`/`alpha` and no `bind=` / `did=tune`. B's C life stamped `bind=adjust` and kept `bind=push`.

## Compare

**A** is the jump, and it **Fail**ed: never-wipe train kept the English bind (PRESS from `push`; argon does not fire). The second life on that dirty store did not stamp C. TM.0.5.6 did this with innate-name stamps; TM.0.6.1 did two English lives on **wiped** S. This slice needed both at once.

**B** shared return **Store-works** on this slice (last-50 0.92; C life did stamp TUNE). Not the jump. Not retuned.

## Audit (not retuned)

- TM.0.5.0–0.6.1 unit tests still pass.
- Default door agent: alias-bind / did-stamp / one-bind / rare-only / revise / here-only **off**.
- TM.0.6.0 `make()` leaves one-bind and rare-only off. TM.0.6.1 `make()` leaves rare-only off.
- Search still computes `has_code`. `domain="dial"` still set. No English lexicon in the agent (`push` / `argon` / `adjust` / `alpha` absent).

## Honest limits

- Stream-first is frozen grammar, not English syntax.
- Tiny closed corpus, not Wikipedia.
- Search is still `{has_code, has_rare}`.
- A's C life committed the second page and did not annotate. Split write was trained only on A stamps; that was enough for TM.0.5.6 innate names and not enough here. Not rescued by raising `n_train` or eval explore.
- Shared return passing here does not rewrite earlier Fail slices.
- Still five acts, innate names, `domain=`. Math is a later life.

## Reproduce

```bash
python tests/test_tm062.py
python tests/test_tm061.py
python tests/test_tm060.py
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
python -m experiments.run_tm062
```
