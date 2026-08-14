# TM.0.5.0 results: no answer integers / shared return

**Date:** 14 August 2026  
**Classification:** **A Store-works** · **B Fail**  
**Run:** `runs/2026-08-14_121951_tm050`

First recipe jump under the locked developmental rule: unread W has **no** place/motor digits. Useful page: “Krypton scrap. Working motor was press.” Held-out: “… tune.” Search cannot use `has_code`. Vname picks a `w*` token (untrained: common word; trained: innate act name). Copy is name→same-named motor (closed body vocabulary). Channel dial. Species prior HOLD. Cortex frozen (`a485b26b…`). `n_forced=0`. Digit-copy off.

## Question

Can find/commit/use work when the unread page does **not** smuggle the answer as integers?

## Predeclared

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; body digits in W; digit-copy restored; synonym lexicon; door world; filed tags; writes from life | same |
| Fail | Untrained PRESS; miss `press` token; after reset not PRESS; C not TUNE; controls still PRESS; swap `idle` still PRESS | Untrained PRESS; A/C miss; empty S solves A; split restored |
| Store-works | Untrained HOLD; free A commits `press`; after ρ reset W gone → PRESS; C TUNE from `tune`; S has `w*` not `n*` | Same without splitting the return |

## Headline

| Check | A split | B shared return |
|-------|---------|-----------------|
| Classification | **Store-works** | **Fail** |
| W body integers | none | none |
| Untrained probe after life | `hold` | `hold` |
| Free A found `press` | **yes** (`w2=press`) | no (`c00` clutter) |
| Trained A after ρ reset, W gone | **`press`** | `hold` |
| Held-out C after reset | **`tune`** (`w3=tune`) | `hold` |
| Exact-match / search-off / use-off / swap | fail / fail / fail / `idle` | fail / fail / fail / `hold` |
| Train last 50 | 0.90 | **0.00** |
| Cortex | unchanged | unchanged (heads never moved) |

## Compare

**A** is the jump: the answer is no longer a digit in the sentence. Free life ranks rare-word pages, commits tokens, learns to copy the innate motor name, survives ρ reset with W gone. Held-out C copies `tune`, not A’s `press`.

**B** shared return **Fail** (last-50 0; clutter; no policy updates). Same credit hole as TM.0.3.x / 0.4.0 B.

## Audit (not retuned)

- TM.0.4.0 digit path still **Store-works** / **Fail** (`use_prose_ints`).
- No synonym table (`push`→PRESS) in the agent.
- S shows `w0=krypton` … `w2=press`, not `n0=0` / `n1=1`.
- Door default `domain="door"` unchanged.

## Honest limits

- Genome still knows the **names of its own acts**. That is body vocabulary, not English NLP. `push` for PRESS is a later English life.
- Still `+3.0` copy, `{has_code, has_rare}` (has_code always false here), five discrete acts.
- Not Wikipedia. Shared return still fails. Self-correction not implemented yet (locked as a later recipe jump).

## Reproduce

```bash
python tests/test_tm050.py
python tests/test_tm040.py
python -m experiments.run_tm050
```
