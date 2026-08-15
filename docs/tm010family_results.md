# TM.0.10.FAMILY results: frozen 0.9.4 vs relation worlds

**Date:** 15 August 2026  
**Regime:** not a recipe jump  
**Run:** `runs/2026-08-15_144235_tm010family`  
**Genome:** frozen TM.0.9.4 (`docs/genome_094.lock`) · delta **0**

## Question

Does the 0.9.4 organism implement a general relation-learning loop, or did we construct five benchmark-specific tricks that happen to compose?

## Headline

| | |
|--|--|
| Worlds | **252/252** solved |
| Developed A–D | **144/144** |
| Hold-out E–G | **108/108** |
| Genome changes | **0** |
| Honest failures | **0** |

12 worlds × 3 births × 7 families. Parallel CPU, 4 workers. No organism edits during the recorded run.

## Families

| Family | Split | Template | Solved |
|--------|-------|----------|--------|
| A | developed | 2 hypotheses, simple reversal | 36/36 |
| B | developed | 3 hypotheses, one irrelevant | 36/36 |
| C | developed | many irrelevant high-support relations | 36/36 |
| D | developed | multiple cues sharing motors | 36/36 |
| E | hold-out | same relation across separated lives | 36/36 |
| F | hold-out | ties that later break | 36/36 |
| G | hold-out | evidence reverses twice | 36/36 |

Hold-out generators were locked before the run. No `use_family_e`.

## Independent measures

| Measure | Rate |
|---------|------|
| Acquisition | 1.000 |
| Survival | 1.000 |
| MATCH | 1.000 |
| Evidence choice | 1.000 |
| Tie handling | 1.000 |
| Revision | 1.000 |
| Reset continuity | 1.000 |
| S necessity | 1.000 |
| Permutation invariance | 1.000 |
| Genome delta | 1.000 |

## Intervention counter

```text
World families attempted: 7
Solved with frozen genome: 7
Required genome changes: 0
Failed honestly: 0
```

## Honest limits

This family stays inside what 0.9.4 already claimed: planted inspectable relations, earned counts, cue in the current stream. It is structural / RNG diversity, not a new skill.

It does **not** show:

- free-life acquisition from unread English W
- composition of two independently learned relations
- math
- a plateau against every future world class

It does show: once the loop (alternatives → survival → MATCH → evidence → choice → revision) is in place, renaming, reordering, adding junk, sharing motors, separating lives, breaking ties, and reversing twice did not require a new recipe.

Composition landed in TM.0.11 as a separate recipe jump (frontier re-entry). Do not fold compose into this freeze.

## Reproduce

```bash
python tests/test_tm010family.py
python -m experiments.run_tm010family --seed 12345 --per-family 12 --births 3 --workers 4
```
