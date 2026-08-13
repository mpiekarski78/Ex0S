# v7 results: native integer tags (no English, not DNA letters)

**Date:** 14 August 2026  
**Classification:** **Store-works**  
**Run:** `runs/2026-08-13_234755_v7`

## What we used

| Layer | Code | Not |
|-------|------|-----|
| Genome | Frozen cortex **seed 1337** + drive/write/select/collect rules | A,T,C,G |
| Senses | 0/1 bits (`Obs.vector`) | English, nucleotides |
| Notes S/W | Integer tags in `.tag` files | Prose, `NOTE:`, Shakespeare |

Example committed fact (`d0.tag`):

```text
# d0
action=2
door=0
requires=1
```

`door=0` is red, `action=2` is `use_key` — **experimenter legend**, not a language the cortex speaks. No four-letter genome.

## Question

> Can the v0 key/door life use v5/v6 collect/select on a **native** store, with no English prior?

## Headline

Empty prior at red door with key: **`open`** (species prior does not know keys).

| Check | Action | Correct? |
|-------|--------|----------|
| A after experience, before ρ reset | use_key | yes |
| A after ρ reset, S kept | use_key | yes |
| **Reload `.tag` files, new agent, ρ empty** | **use_key** | **yes** |
| B foil (blue life) after ρ reset | open | no |
| disable-S after ρ reset | open | no |
| disable-S before ρ reset | use_key | yes (session) |
| Collect commit, unmount W (13 files) | use_key | yes (only `d0.tag` in S) |
| Peek then unmount W | open | no (not memory) |
| Collect off, W mounted | open | no |
| Dump-all 13 tags | **open** | no (clutter wins) |
| Delete S | open | no |
| English prose in tags | — | **false** |
| Weights unchanged | — | yes |

W has 13 `.tag` files. Select matches `door=0`. Dump-all applies every `action=1` (open) bias and prefers `open`. Same lesson as v4: growth needs select.

## What this means

The innate machine does not speak English and does not use DNA letters as eyes. It uses **bits and integer tags**. The genome is the frozen seed + rules. World knowledge is a file of numbers. Collect still copies one matching record; a glance at W is not a life.

v1–v6 remain the BDH-language comparison. v7 is the biological-direction substrate.

## Honest limits

- Tags are still a tiny designed code, not a learned language.
- Retrieve is tag→action (v0), not parsing.
- `0`/`1`/`2` are not nucleotides; do not read this as a DNA simulation.

## Reproduce

```bash
python tests/test_v7.py
python -m experiments.run_v7
```
