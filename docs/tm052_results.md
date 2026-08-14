# TM.0.5.2 results: unread W unnamed / shared return

**Date:** 14 August 2026  
**Classification:** **A Store-works** · **B Fail**  
**Run:** `runs/2026-08-14_124528_tm052`

Second slice of “use the file as text.” Unread W has **no** motor name (`press` / `tune` absent) and **no** digits. Useful page: “Krypton scrap. Working notes from the bench.” Held-out: helium scrap. The body stamps the act it just did onto a rare committed note (`w7=press`). After ρ reset, W gone, vname copies that innate name. Channel dial. Species prior HOLD. Cortex frozen (`a485b26b…`). `n_forced=0`. Digit-copy off. v9 `write_from_events` off.

## Question

Can find/commit/use work when the unread page does **not** smuggle the answer as a motor name?

## Predeclared

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; motor name still in W; digits; digit-copy; synonym lexicon; door world; filed tags; v9 writes from life | same |
| Fail | Untrained PRESS or already stamps; miss stamp; after reset not PRESS; C not TUNE; annotate-off / clutter-only still PRESS | Untrained PRESS; A/C miss; empty S solves A; split restored |
| Store-works | Untrained HOLD; free A commits krypton and stamps `press`; after ρ reset W gone → PRESS; C TUNE from stamped `tune` | Same without splitting the return |

## Headline

| Check | A split | B shared return |
|-------|---------|-----------------|
| Classification | **Store-works** | **Fail** |
| W motor names / body ints | none / none | none / none |
| Untrained probe after life | `hold` (n_annotated 0) | `hold` |
| Free A stamp | **yes** (`p99` `w7=press`) | no (`c00` clutter) |
| Trained A after ρ reset, W gone | **`press`** | `hold` |
| Held-out C after reset | **`tune`** (`p98` `w7=tune`) | `hold` |
| Exact-match / search-off / vname-off / annotate-off / clutter-only | `hold` / `hold` / `hold` / `hold` / `hold` | all `hold` |
| Train last 50 | 0.36 | **0.00** |
| Cortex | unchanged | unchanged (heads never moved) |

## Compare

**A** is the jump: the answer is no longer a word in the unread page. Free life ranks rare-word scraps, commits krypton, succeeds at PRESS, stamps the body’s act name onto that note, survives ρ reset with W gone. Held-out C stamps `tune` after a TUNE in life, not A’s `press`.

**B** shared return **Fail** (last-50 0; clutter; no policy updates). Same credit hole as TM.0.3.x–0.5.1 B.

## Audit (not retuned)

- TM.0.5.0 still **Store-works** / **Fail** with the name in the page.
- TM.0.5.1 still **Store-works** / **Fail** with revise.
- No synonym table (`push`→PRESS) in the agent.
- S shows `w0=krypton` … `w7=press`, not `n*` digits and not a motor name copied from W.
- Door default `domain="door"` unchanged. `use_event_annotate` default off.
- Annotate-off and clutter-only stay HOLD: W is still load-bearing (rare page), and the stamp is load-bearing (not name-in-the-page).

## Honest limits

- Genome still knows the **names of its own acts**. Stamping `press` is body vocabulary, not English NLP. `push` for PRESS is a later English life.
- Stamp only onto a **rare** committed note (same rarity bit as search). Clutter words are common; krypton is not.
- Copy of a selected innate act name is frozen grammar (a non-act token does not bias the motor). The learned heads are search / write / vname.
- A split is three returns: find the rare page / stamp the act / copy after reset. Shared return still fails.
- Experience life does not abort on lucky PRESS (same as TM.0.5.1). Commit does not replace a stamped note with the unread original.
- Still five discrete acts. Not Wikipedia.

## Reproduce

```bash
python tests/test_tm052.py
python tests/test_tm051.py
python tests/test_tm050.py
python -m experiments.run_tm052
```
