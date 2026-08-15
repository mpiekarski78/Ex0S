# TM.0.9.BOX results: boxed-policy leakage control

**Date:** 15 August 2026  
**Compatible label:** separation **Control Fail** (all 3 seeds)  
**Run:** `runs/2026-08-15_103142_tm091box`

Not a recipe jump. Frozen TM.0.9.1 genome and development protocol (both delta 0). Cortex hash unchanged (`a485b26b…`). Parallel CPU workers only. No CUDA. No organism edits.

Historical BOX stays frozen. The B readout has **no `flim` cue** — the cue is which projected note is mounted. That is the leakage test. Do not mutate this probe after MATCH lands, and do not require this battery to turn green.

## Question

Do world-specific `bind→did` associations live in S, or leak into boxed weights?

## Measures

| Measure | Result |
|---------|--------|
| World-fact leakage | **Not observed**, 3/3 seeds |
| Counterfactual donor control | **Pass**, 3/3 |
| Neutral relevance control | **Fail**, 3/3 |
| Cross-world use-S, evaluable stores | **Pass**, 2/2 |
| W3 acquisition robustness | **2/3** |

Compatible protocol label remains **Control Fail** because neutrals blindly copy `did`. That label must not hide the leakage result.

## Lethal pair (recorded table)

Station B (innate HOLD). Mechanically projected, field-canonical donor stores. Same birth for P1/P2 within each seed. No environmental nonce in the observation.

| Policy | Empty | Neutral PRESS | Neutral TUNE | S1 `flim→PRESS` | S2 `flim→TUNE` |
|--------|-------|---------------|--------------|-----------------|----------------|
| P0 | HOLD | HOLD | HOLD | HOLD | HOLD |
| P1 | HOLD | **PRESS** | **TUNE** | **PRESS** | **TUNE** |
| P2 | HOLD | **PRESS** | **TUNE** | **PRESS** | **TUNE** |

All three seeds matched this pattern on the separation battery.

## What the crossover showed

- **World-fact leakage:** P1 and P2 HOLD on empty S. Boxed weights do **not** reconstruct `flim→PRESS` vs `flim→TUNE`. The audit’s Confound signature did not fire.
- **Donor control:** `Pi + S1 → PRESS`, `Pi + S2 → TUNE` for both P1 and P2. Training history does not override the projected store.
- **Neutral relevance:** `wibble→press` and `tork→tune` both fire their `did` for every trained Pi. Blind copy when any bind→did note is present. That is why the compatible label is Control Fail, not Confound.
- **Genome / protocol:** locks match. `make()` untouched.

## Transfer is not acquisition

| Seed | S3 (`blen`) | Cross-world use-S |
|------|-------------|-------------------|
| 12345 | acquired | Pass (P1/P2 use S3; P3 uses S1/S2) |
| 12346 | acquired | Pass |
| 12347 | **missed** (S n=2; projection empty) | **Unevaluable** — no valid S3 |

Seed 12347 is an acquisition miss, not a transfer miss. Do not raise `n_train` to force 3/3.

## Claim earned

> World-specific action associations were acquired during life, survived ρ reset, were not recoverable from the trained policy on empty S, and counterfactually changed behavior when a mechanically projected, field-canonical donor store was swapped.

Not earned: the organism knows which fact in S applies. Indiscriminate use of any projected bind→did note still contaminates the no-cue neutral baseline. That is a MATCH problem, not a leakage problem. Historical BOX does not present a current antecedent, so it is the wrong battery for MATCH.

## Next (not this control)

TM.0.9.2: first-class antecedent MATCH. A stored `X→action` may steer only when `X` is in the current observation. Independent-support work waits until applicability is solved. Cue-bearing BOX-MATCH is a new audit arm — do not overwrite this experiment.

## Reproduce

```bash
python tests/test_tm091box.py
python tests/test_tm091.py
python tests/test_tm050.py
python -m experiments.run_tm091box --seeds 12345 12346 12347 --workers 3
```
