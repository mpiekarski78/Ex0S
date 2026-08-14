# TM.0.4.0 results: leave the door world / shared return

**Date:** 14 August 2026  
**Classification:** **A Store-works** · **B Fail**  
**Run:** `runs/2026-08-14_120627_tm040`

Same free-life prose procedure as TM.0.3.2, but the world is **not** key/door.  
`ChannelDialWorld`: three channels, five motors (IDLE/PRESS/HOLD/TUNE/FLIP).  
Correct: A→PRESS, B→HOLD, C→TUNE. Held-out C place code equals PRESS — copying place fails.  
Species prior prefers HOLD (wrong on A and C). Cortex seed 1337, **5** action logits (new SHA256 vs door stack). No NOTE-copy. `n_forced=0`.

## Question

Can find/commit/use from pure prose survive leaving the four-act door toy — same boxed heads, new place/motor codes, held-out place-copy trap?

- Genome digit scan (not English). Heading digits excluded.
- Search `{has_code, has_rare}`; vname `{is_code, val_common}`.
- **A.** Split: found place+motor ints in S during life / probe after ρ reset.
- **B.** One shared return.

## Predeclared

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; door world restored; filed `action=`/`where=`; `.tag` W; writes from life; forced curriculum; empty S already TUNE on C | same |
| Fail | Untrained already PRESS on A; free A life misses pair; after reset not PRESS; held-out C not TUNE; exact match / freeze search/vname/use still PRESS; swap IDLE still PRESS | Untrained PRESS; A miss; C miss; empty S solves A; split restored |
| Store-works | Untrained HOLD; free A commits `n0=0`,`n1=1`; after ρ reset W gone → PRESS; held-out C TUNE not place-copy; S has `n*` | Same without splitting the return |

Do not restore the door toy or filed `action=` to rescue a plot.

## Headline

| Check | A split | B shared return |
|-------|---------|-----------------|
| Classification | **Store-works** | **Fail** |
| World | channel_dial | channel_dial |
| Untrained probe after life | `hold` | `hold` |
| Free A found place+PRESS | **yes** (`p99` `n0=0` `n1=1`) | no (`c01` clutter) |
| Trained A after ρ reset, W gone | **`press`** | `press` (clutter place-copy) |
| Held-out C after reset | **`tune`** (`p98` `n0=1` `n1=3`) | `idle` |
| Exact-match / search-off / use-off | fail / fail / fail | fail / fail / fail |
| Name-swap (IDLE prose) | `idle` | `press` |
| Train last 50 | 0.88 | 0.92 |
| Cortex | unchanged (`a485b26b…`) | unchanged (same hash) |

## Compare

**A** is the jump: left the door room. Free life on a dial bench ranks prose, commits anonymous ints, copies the non-place motor, survives ρ reset with W gone. Held-out C must TUNE — place code is PRESS, so place-copy fails.

**B** shared return **Fail**: last-50 high from accidental PRESS via clutter place=`1`, but never commits the useful A/C pages; held-out C stays wrong. Same credit hole as TM.0.3.x B — not a new plot.

## Audit (not retuned)

- Door default `domain="door"` unchanged; TM.0.3.2 still **Store-works** / **Fail**.
- Dial cortex hash differs (5 actions) — expected; still frozen seed 1337.
- Species prior HOLD so empty S cannot look like Store-works on A.
- Clutter may plant place=`1` (channel C / PRESS code) as the same structural accident door clutter planted place=`2`=USE_KEY; per-file `found_a_pair` still requires `{0,1}` on one page (only `p99`).
- No `KeyDoorWorld` / `probe_red` in the TM.0.4.0 experiment.

## Honest limits

- Still digit scan + anonymous `n*`, not English.
- Still genome `logits[int] += 3.0` and `{has_code, has_rare}`.
- Five acts on a three-channel bench — not Wikipedia.
- Shared return still fails the held-out channel.

## Reproduce

```bash
python tests/test_tm040.py
python tests/test_tm032.py
python -m experiments.run_tm040
```
