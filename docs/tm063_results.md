# TM.0.6.3 results: new-here stamp / shared return

**Date:** 14 August 2026  
**Classification:** **A Store-works** · **B Store-works**  
**Run:** `runs/2026-08-14_181124_tm063`

Recipe jump: **a new station gets an unmarked rare page.** TM.0.6.2 kept `bind=push` through never-wipe train, then a C life TUNEd (`opened`) and did not stamp — the write head trained on A did not generalize. Genome: if S already names some other station, a success here stamps an unmarked rare note (commit one if needed). Probe still HOLDs at an unnamed station (species prior stays on greedy). After train: dirty S **PRESS** / foil C **HOLD**. C life on that S: A **PRESS** kept, C **TUNE** from `bind=adjust`. Wipe-between loses A. Cortex frozen (`a485b26b…`). `n_forced=0`. `n_train` was not raised. `use_stamp_new_here` default **off**.

Skipped on purpose: dropping `has_code`, removing `domain=`, math, solving B (B happened to pass; not retuned), turning on 0.5.9 correct flags.

## Question

Can a growing English store take a second station as a new unmarked page, or does every new place need the experimenter to wipe S or retune the write head?

## Predeclared

| ID | A | B |
|----|---|----|
| Confound | Cortex moves; unique-rare needle; `push` in the agent; drop `has_code` / `domain=`; revise/here-only on; probe TUNE without a C bind; train still wipes | same |
| Fail | Untrained PRESS; train S not PRESS; C life loses A or misses TUNE; wipe-between still PRESS; bind=`argon`; nonce PRESS | Untrained PRESS; A miss; C miss; nonce PRESS |
| Store-works | Untrained HOLD; never-wipe train A PRESS / C HOLD from `push`; C life on that S A PRESS / C TUNE from `adjust`; wipe-between A HOLD; nonce HOLD | Same without splitting the return |

## Headline

| Check | A split | B shared return |
|-------|---------|-----------------|
| Classification | **Store-works** | **Store-works** |
| Untrained A / foil C | `hold` / `hold` | `hold` / `hold` |
| After train, dirty S: A / foil C | **`press` / `hold`** | **`press` / `hold`** |
| C life on dirty S: A / C | **`press` / `tune`** | **`press` / `tune`** |
| Wipe-between A | **`hold`** | **`hold`** |
| Bind-off A | **`hold`** | **`hold`** |
| Nonce-only A | **`hold`** | **`hold`** |
| Bind-all nonce A | **`press`** | **`press`** |
| Train last 50 | 0.92 | 0.92 |
| Cortex | unchanged | unchanged |

C note: `bind=adjust`, `did=tune`, `alpha` kept, not bound. Train note `bind=push` / `cha` kept.

## Compare

**A** is the jump: TM.0.6.2 committed `p98` and left it unstamped because write was an A-head. New-here stamp is frozen grammar (a new place is a new unmarked page), not a raised `n_train`. Probe still HOLD without a C bind.

**B** shared return **Store-works** on this slice (last-50 0.92). Not the jump. Not retuned.

## Audit (not retuned)

- TM.0.5.0–0.6.2 unit tests still pass.
- Default door agent: stamp-new-here / one-bind / rare-only / revise **off**.
- TM.0.6.0–0.6.2 `make()` leave stamp-new-here off. TM.0.6.2 A remains Fail without it.
- Search still computes `has_code`. `domain="dial"` still set. No English lexicon in the agent.

## Honest limits

- Stream-first is still frozen grammar, not English syntax.
- Tiny closed corpus, not Wikipedia.
- Search is still `{has_code, has_rare}`.
- New-here stamp is innate on success, not a learned “when” for the second place.
- Shared return passing here does not rewrite earlier Fail slices.
- Still five acts, innate names, `domain=`. Math is a later life.

## Reproduce

```bash
python tests/test_tm063.py
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
python -m experiments.run_tm063
```
