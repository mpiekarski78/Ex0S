# v11 results: select among notes the agent wrote

**Date:** 14 August 2026  
**Classification:** **Store-works**  
**Run:** `runs/2026-08-14_090439_v11`

## Question

> After two **free** lives, can the agent **pick** the matching authored note, or does it have to dump S?

v4 showed dump-all fails on experimenter-written `.md` files. v11 uses notes the agent **authored** (`n_forced = 0`, W has no answers). Same frozen select rule: match `door=`. Dump-all applies every `action=` bias.

## Predeclared

| ID | If |
|----|----|
| Confound | Cortex hash moves; disable-S still correct; answer in W; `n_forced > 0`; probe explores |
| Fail | Only one file in S; select red or green wrong; dump-all as good as select on **both** probes |
| Store-works | Both files authored; select red `use_key` and green `wait`; dump-all fails at least one probe; empty S / disable-S fail |

## Headline

S after red life then green life: **`d0.tag` and `d2.tag`**. Cortex unchanged. Policy changed (write skill from v10 training).

| Check | Select | Dump-all |
|-------|--------|----------|
| Red with key | **`use_key`** | **`wait`** (wrong) |
| Green | **`wait`** | `wait` (accidentally right) |
| Empty S | `open` | — |
| disable-S | `open` | — |
| n_forced | **0 / 0** | — |

Dump-all on red waits because it also applies `d2.tag` (`action=0`). Select ignores that file at the red door. Dump-all matching select on green does not rescue dump: N is already large enough to break red.

## What this means

v10: live, then write.  
v11: **live twice, then pick.**

Growth of S needs a select rule even when the notes are the agent’s own life. Dumping the pile mixes lives. That is the index, still tag match, still not RAG.

## Honest limits

- Two files, not a wiki. Dump fails at this N; do not claim a vector index is required.
- Select is hardcoded `door=` match, not a learned ranker.
- Dump-all green is still `wait` by accident (WAIT logit wins the mix).
- Write skill and free exploration are inherited from v10.

## Reproduce

```bash
python tests/test_v11.py
python tests/test_v10.py
python -m experiments.run_v11
```
