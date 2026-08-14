# TM.0.6.4 results: English find without unique rare / shared return

**Date:** 14 August 2026  
**Classification:** **A Fail** · **B Store-works**  
**Run:** `runs/2026-08-14_185628_tm064`

Recipe jump: **English find without a unique rare token.** Same never-wipe one-bind + new-here recipe as TM.0.6.3, but Open W has three hapax clutter pages (`xenon` / `neon` / `krypton` on `c08`–`c10`). Not `argon` — that is already the distractor on `p99`. `has_rare` is no longer a unique pointer at the useful page. After never-wipe train the dirty store (6 files) still probes **PRESS** / foil C **HOLD**, and a C life on that S is A **PRESS** / C **TUNE** from `bind=adjust`. The English bar fails: search stamped the clutter hapax as acts (`bind=xenon`, `bind=neon`, `bind=krypton`) as well as `bind=push`. Closed-lexicon clutter-only (no hapax) stays HOLD. Cortex frozen (`a485b26b…`). `n_forced=0`. `n_train` was not raised.

Skipped on purpose: dropping `has_code`, removing `domain=`, math, solving B (B happened to pass the motor bar; not retuned), turning on 0.5.9 correct flags, adding a ranker to prefer `p99`.

## Question

Can English search still bind the useful page word when several unread documents are distinctive, or does the recipe need one unique rare needle (or a new ranker)?

## Predeclared

| ID | A | B |
|----|---|----|
| Confound | Cortex moves; useful page is still the only rare token; `push` in the agent; drop `has_code` / `domain=`; revise/here-only on; stamp-new-here off; argon used as clutter hapax | same |
| Fail | Untrained PRESS; train S not PRESS; C life loses A or misses TUNE; wipe-between still PRESS; bind=`xenon`/`neon`/`krypton`/`argon`; nonce PRESS; closed clutter-only PRESS | Untrained PRESS; A miss; C miss; nonce PRESS |
| Store-works | Multi-rare English W; untrained HOLD; never-wipe train A PRESS / C HOLD from `push` only; C life on that S A PRESS / C TUNE from `adjust`; wipe-between A HOLD; nonce HOLD; closed clutter-only HOLD | Same without splitting the return |

## Headline

| Check | A split | B shared return |
|-------|---------|-----------------|
| Classification | **Fail** | **Store-works** |
| Untrained A / foil C | `hold` / `hold` | `hold` / `hold` |
| After train, dirty S: A / foil C | `press` / `hold` | **`press` / `hold`** |
| C life on dirty S: A / C | `press` / `tune` | **`press` / `tune`** |
| Wipe-between A | `hold` | `hold` |
| Bind-off A | `hold` | `hold` |
| Nonce-only A | `hold` | `hold` |
| Bind-all nonce A | `press` | `press` |
| Closed clutter-only A | `hold` | `hold` |
| Rare clutter pages | 3 | 3 |
| Train S n files | 6 | 8 |
| Train last 50 | 0.94 | 0.94 |
| Cortex | unchanged | unchanged |

Train note `p99`: `bind=push`, `did=press`, `argon` kept. Also `c08` `bind=xenon`, `c09` `bind=neon`, `c10` `bind=krypton` — all `did=press` at `cha`. C note: `bind=adjust`, `did=tune`, `alpha` kept, not bound. `bind=push` kept.

## Compare

**A** is the jump: TM.0.5.7 did this for nonce scraps (stamping any rare page with `press` works). English must still bind `push`, not a clutter hapax. `{has_code, has_rare}` cannot prefer `p99` over xenon. The motors still work; the English bind is not selective. Honest Fail. Do not restore unique-rare. Do not add a ranker in this slice.

**B** shared return **Store-works** on the motor bar (last-50 0.94). Not the jump. Not retuned. Same dirty S also bound the clutter hapax.

## Audit (not retuned)

- TM.0.5.0–0.6.3 unit tests still pass.
- Default door agent: stamp-new-here / one-bind / rare-only / revise **off**.
- TM.0.6.0–0.6.2 `make()` leave stamp-new-here off. TM.0.6.3/0.6.4 turn it on.
- Search still computes `has_code`. `domain="dial"` still set. No English lexicon in the agent. No new agent flag.

## Honest limits

- Stream-first is still frozen grammar, not English syntax.
- Tiny closed corpus, not Wikipedia.
- Search is still `{has_code, has_rare}` — uniqueness gone, ranking not invented.
- Shared return passing here does not rewrite this Fail or earlier Fail slices.
- Still five acts, innate names, `domain=`. Math is a later life.

## Reproduce

```bash
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
python -m experiments.run_tm064
```
