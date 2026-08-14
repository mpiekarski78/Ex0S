# TM.0.6.5 results: concurrent bind / shared return

**Date:** 14 August 2026  
**Classification:** **A Store-works** · **B Fail**  
**Run:** `runs/2026-08-14_190612_tm065`

Recipe jump: **stamp the page in play, then block extra hapax at this station.** Same English multi-rare W as TM.0.6.4 (`xenon` / `neon` / `krypton` plus `p99`). TM.0.6.4 bound every rare page as PRESS. Genome: the CS is the attended file when the body succeeds; once this station has a bind, do not add a second CS. After never-wipe train, only `p99` is bound (`bind=push`, `did=press`); clutter hapax sit unmarked. Dirty S **PRESS** / foil C **HOLD**. C life on that S: A **PRESS** kept, C **TUNE** from `bind=adjust`. Wipe-between loses A. Cortex frozen (`a485b26b…`). `n_forced=0`. `n_train` was not raised. `use_block_here` default **off**.

Skipped on purpose: dropping `has_code`, removing `domain=`, math, solving B, restoring unique-rare, turning on 0.5.9 here-only/revise, adding a ranker.

## Question

Can English search bind one concurrent page when several unread documents are distinctive, or does every hapax become the act?

## Predeclared

| ID | A | B |
|----|---|----|
| Confound | Cortex moves; unique-rare restored; `push` in the agent; drop `has_code` / `domain=`; revise/here-only on; stamp-new-here off; argon as clutter hapax | same |
| Fail | Untrained PRESS; train S not PRESS; C life loses A or misses TUNE; wipe-between still PRESS; bind=`xenon`/`neon`/`krypton`/`argon`; nonce PRESS | Untrained PRESS; A miss; C miss; nonce PRESS |
| Store-works | Multi-rare English W; one CS here from the page in play; untrained HOLD; train A PRESS / C HOLD from `push`; C life A PRESS / C TUNE from `adjust`; wipe-between A HOLD; nonce HOLD | Same without splitting the return |

## Headline

| Check | A split | B shared return |
|-------|---------|-----------------|
| Classification | **Store-works** | **Fail** |
| Untrained A / foil C | `hold` / `hold` | `hold` / `hold` |
| After train, dirty S: A / foil C | **`press` / `hold`** | `press` / `hold` |
| C life on dirty S: A / C | **`press` / `tune`** | `press` / **`hold`** |
| Wipe-between A | **`hold`** | `hold` |
| Bind-off A | **`hold`** | `hold` |
| Nonce-only A | **`hold`** | `hold` |
| Bind-all nonce A | **`press`** | `press` |
| Closed clutter-only A | `hold` | `hold` |
| Train S binds | **`push` only** | `neon` |
| Train last 50 | 0.92 | **0.96** |
| Cortex | unchanged | unchanged |

A train note `p99`: `bind=push`, `did=press`, `argon` kept. Clutter hapax files are in S unmarked. C note: `bind=adjust`, `did=tune`, `alpha` kept. `bind=push` kept.

## Compare

**A** is the jump: uniqueness stays gone. Search still `{has_code, has_rare}`. The write no longer stamps leftover rares. Split `r_find` taught the find head to attend `p99`; in-hand stamped that page; blocking kept xenon/neon/krypton from becoming PRESS. TM.0.6.4 without this genome bound all four.

**B** shared return **Fail**: first CS was `bind=neon`; C life did not TUNE. Last-50 **0.96** on the first fact. Not the jump. Not retuned. Split credit was load-bearing for attending `push` in time.

## Audit (not retuned)

- TM.0.5.0–0.6.4 unit tests still pass.
- Default door agent: block-here / stamp-new-here / one-bind / rare-only / revise **off**.
- TM.0.6.0–0.6.4 `make()` leave block-here off. TM.0.6.4 A remains Fail without it.
- Search still computes `has_code`. `domain="dial"` still set. No English lexicon in the agent.

## Honest limits

- Stream-first is still frozen grammar, not English syntax.
- Tiny closed corpus, not Wikipedia.
- Search is still `{has_code, has_rare}`. Split credit still teaches which page to attend.
- Blocking is first-CS-wins: a shared-return life that attends neon first stays neon.
- S still accumulates unmarked clutter (here-only/revise stayed off).
- Still five acts, innate names, `domain=`. Math is a later life.

## Reproduce

```bash
python tests/test_tm065.py
python tests/test_tm064.py
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
python -m experiments.run_tm065
```
