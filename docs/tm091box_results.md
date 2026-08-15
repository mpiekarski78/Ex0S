# TM.0.9.BOX results: boxed-policy leakage control

**Date:** 15 August 2026  
**Separation:** **Control Fail** (all 3 seeds)  
**Transfer:** **Fail** (2/3 Pass; seed 12347 W3 acquisition missed `blen`)  
**Run:** `runs/2026-08-15_103142_tm091box`

Not a recipe jump. Frozen TM.0.9.1 genome and development protocol (both delta 0). Cortex hash unchanged (`a485b26b…`). Parallel CPU workers only. No CUDA. No organism edits.

## Question

Do world-specific `bind→did` associations live in S, or leak into boxed weights?

## Lethal pair (recorded table)

Station B (innate HOLD). Mechanically projected, field-canonical donor stores. Same birth for P1/P2 within each seed.

| Policy | Empty | Neutral PRESS | Neutral TUNE | S1 `flim→PRESS` | S2 `flim→TUNE` |
|--------|-------|---------------|--------------|-----------------|----------------|
| P0 | HOLD | HOLD | HOLD | HOLD | HOLD |
| P1 | HOLD | **PRESS** | **TUNE** | **PRESS** | **TUNE** |
| P2 | HOLD | **PRESS** | **TUNE** | **PRESS** | **TUNE** |

All three seeds matched this pattern on the separation battery.

## Headline

| Axis | Result | Meaning |
|------|--------|---------|
| Separation | **Control Fail** | Empty S stays HOLD — no world-fact leak into P. Counterfactual S1/S2 follows the **donor**. Neutrals fire a **world-independent** copy of `did`. |
| Transfer | **Fail** | Seeds 12345/12346 Pass. Seed 12347: W3 never bound `blen` (S n=2), so S3 projection empty. |

## What the crossover showed

- **Empty S:** P1 and P2 HOLD. Boxed weights do **not** reconstruct `flim→PRESS` vs `flim→TUNE`. The audit’s Confound signature did not fire.
- **Donor swap:** `Pi + S1 → PRESS`, `Pi + S2 → TUNE` for both P1 and P2. Training history does not override the projected store.
- **Neutrals:** `wibble→press` and `tork→tune` both fire their `did` for every trained Pi. Blind copy when any bind→did note is present. That is **Control Fail**, not Confound.
- **Genome / protocol:** locks match. `make()` untouched.

## Transfer (separate axis)

| Seed | Transfer | Note |
|------|----------|------|
| 12345 | Pass | P1/P2 use S3; P3 uses S1/S2 |
| 12346 | Pass | same |
| 12347 | Fail | W3 acquisition missed `blen`; not a leakage result |

## Claim earned

> World-specific action associations were acquired during life, survived ρ reset, were not recoverable from the trained policy on empty S, and counterfactually changed behavior when a mechanically projected, field-canonical donor store was swapped. Indiscriminate use of any projected bind→did note still contaminates the neutral baseline.

Not earned: full Store-works (neutrals must HOLD). Not Confound: facts did not leak into P as a world-correlated map.

## Reproduce

```bash
python tests/test_tm091box.py
python tests/test_tm091.py
python tests/test_tm050.py
python -m experiments.run_tm091box --seeds 12345 12346 12347 --workers 3
```
