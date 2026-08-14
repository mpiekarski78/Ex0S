# TM.0.6.9 results: find-novel without a unique two-rare pair / shared return

**Date:** 14 August 2026  
**Classification:** **A Fail** · **B Store-works**  
**Run:** `runs/2026-08-14_224508_tm069`

Recipe jump: **same find-novel as TM.0.6.8, but several clutter pages also match the novel-count of `p99`/`p98`.** Useful pages score 3 because `# p99`/`# p98` tokenize to rare `p`; two body hapax still lose, so late clutter (`c08`/`c09`/`c10`) gets a third hapax (`radon`+`lithium`, `cesium`+`nickel`, `cobalt`+`quartz` on top of `xenon`/`neon`/`krypton`). Train S is **1 file**, `bind=neon` on `c09`, n_revised=0, probe **PRESS** / foil C **HOLD**. A later C life TUNEs from `c08` `bind=xenon`, not from `adjust`. Wipe-between loses A. Cortex frozen (`a485b26b…`). `n_forced=0`. `n_train` was not raised. Find-novel default **off**. No new agent flag.

Skipped on purpose: dropping `has_code`, removing `domain=`, math, ranking `p98` by name, restoring unique-rare, restoring leftover W walk, raising `MAX_TRAIN_S_FILES`, adding a third-hapax ranker.

## Question

Can find-novel still pick the useful English page when several unread clutter pages would add as many rare tokens as `p99`/`p98`, or was TM.0.6.8 unique-pair renamed?

## Predeclared

| ID | A | B |
|----|---|----|
| Confound | Cortex moves; unique two-rare pair restored; `push` in the agent; drop `has_code` / `domain=`; leftover walk kept; ranker for `p98`; argon as clutter hapax; n_train raised | same |
| Fail | Untrained PRESS; train S not PRESS; train S still >4 files; C life loses A or misses TUNE; wipe-between still PRESS; bind=`xenon`/`neon`/`krypton`/`radon`/`lithium`/`cesium`/`nickel`/`cobalt`/`quartz`/`argon`; nonce PRESS | Untrained PRESS; A miss; C miss; n>4; nonce PRESS |
| Store-works | Two-rare-clutter English W; train S small; A PRESS / C HOLD from `push`; C life A PRESS / C TUNE from `adjust`; wipe-between A HOLD; nonce HOLD | Same without splitting the return |

## Headline

| Check | A split | B shared return |
|-------|---------|-----------------|
| Classification | **Fail** | **Store-works** |
| Untrained A / foil C | `hold` / `hold` | `hold` / `hold` |
| After train, S: A / foil C | **`press` / `hold`** | **`press` / `hold`** |
| C life on that S: A / C | `press` / `tune` | **`press` / `tune`** |
| Wipe-between A | **`hold`** | **`hold`** |
| Train S n files | **1** | **1** |
| n_revised train | 0 | 0 |
| Train S binds | **`neon`** | `xenon` |
| C life binds | `neon` + **`xenon`** | `xenon` + `neon` |
| Train last 50 | 0.94 | 0.92 |
| Cortex | unchanged | unchanged |

A train note is only `c09`: `bind=neon`, `did=press`. C added `c08` `bind=xenon` `did=tune`. Closed clutter-only HOLD. Nonce HOLD.

## Compare

**A** is the jump: TM.0.6.8 Store-works because `p99`/`p98` were the only pages with the max novel-count (two body hapax plus heading `p`). Once three late clutter pages also score 3, find-novel keeps `c08`/`c09`/`c10` with them. Train attended `c09` / `neon`, not `p99` / `push`. C life attended `c08` / `xenon`, not `p98` / `adjust`. Motors still fire from `did=`. The English two-station bar fails on the bind. Do not restore unique-pair. Do not rank `p98`.

**B** shared return **Store-works** on the motor bar (n=1, PRESS/TUNE from `xenon`+`neon`, last-50 0.92). Not the jump. Not retuned.

## Audit (not retuned)

- TM.0.5.0–0.6.9 unit tests still pass.
- Default door agent: find-novel / in-hand new-here / revise / here-only / block-here / stamp-new-here / one-bind **off**.
- TM.0.6.0–0.6.8 `make()` leave this W off. TM.0.6.8 on the one-hapax pile still keeps only `p99`/`p98`; it Confounds if two-rare clutter is smuggled onto that slice.
- Search still computes `has_code`. `domain="dial"` still set. No English lexicon in the agent.

## Honest limits

- Stream-first is still frozen grammar, not English syntax.
- Tiny closed corpus, not Wikipedia.
- Search is still `{has_code, has_rare}` after the novel-count filter. Heading `# p99`/`# p98` is a hidden third rare token (`p`). Two body hapax do not compete; three do.
- Split credit still teaches A to attend whatever page find held; it does not know `push` from `neon`.
- Shared return passing here does not rewrite this Fail or earlier Fail slices.
- Still five acts, innate names, `domain=`. Math is a later life.

## Reproduce

```bash
python tests/test_tm069.py
python tests/test_tm068.py
python tests/test_tm067.py
python tests/test_tm066.py
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
python -m experiments.run_tm069
```
