# TM.0.11 results: COMPOSE / motor bar

**Ex0S:** **0.0.002**  
**Date:** 15 August 2026  
**Classification:** **A Store-works** · **B Fail**  
**Run:** `runs/2026-08-15_145626_tm011`

Recipe jump: **COMPOSE.** One primitive — a chosen non-motor consequent becomes the next MATCH frontier. Visited fact ids excluded. No hop cap. No accumulated stream. Act-local derived state. No shortcut write into S. Cortex frozen. `policy.py` unchanged. `n_train` not raised.

## Question

Can two independently acquired relations jointly determine a novel motor when that exact combination was never stored or reinforced?

## Headline

| Check | A COMPOSE | B motor bar |
|-------|-----------|-------------|
| Classification | **Store-works** | **Fail** |
| Main `X→Y`, `Y→M1`, cue X | **M1** (hops=2) | — |
| Broken / wrong-first | **HOLD** | — |
| Wrong second / upstream donor | follows donor S | — |
| Downstream donor swap | PRESS / TUNE | — |
| Irrelevant support=1000 | dropped by MATCH | — |
| Direct `X→motor` before/after | **absent** | — |
| S hash across use | **stable** | — |
| No residue across acts | **HOLD** | — |
| Wipe / ρ reset | HOLD / keep | — |
| Permuted seeds | **3/3** | — |
| BOX-MATCH (0.9.2 make) | **Store-works** 3/3 | — |

## Claim (conservative)

Two independently acquired relations can be composed at use time to produce behavior that was never directly stored or reinforced, without materializing a shortcut relation in persistent memory.

Not reasoning. Not symbolic intelligence. The qualitative jump is:

```text
cue → stored relation → derived frontier → stored relation → motor
```

instead of only:

```text
cue → stored relation → motor
```

## Honest limits

- First battery is 2-hop. Depth and branching wait for a frozen-compose family.
- 0.9.4 / 0.10.FAMILY `make()` stays compose-off.
- Historical BOX neutrals remain frozen 0.9.1 behavior.
- Do not add confidence, decay, or hop-counted flags because this was green.
- Filenames use `tm011compose` so they do not collide with TM.0.1.1 (`tm011`).

## Reproduce

```bash
python tests/test_tm011compose.py
python tests/test_tm094.py
python tests/test_tm092.py
python -m experiments.run_tm011compose
```
