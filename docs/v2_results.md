# v2 results: raw retrieve, no NOTE-copy

**Date:** 14 August 2026  
**Classification:** **Trace-only**  
**Checkpoint:** `checkpoints/prior_plain.pt` (gitignored; `python -m experiments.train_prior --plain`)  
**NOTE-follow acc (must stay low):** **0.025** (chance ≈ 1/26; copy was not taught)  
**Run:** `runs/2026-08-13_224818_v2`

## Question

> If S is retrieved as ordinary text — and the frozen LM was **never** trained to copy `NOTE:` lines — do facts still survive ρ reset?

This is the harder “real model using S” test. v1 Store-works was allowed to teach a retrieve format. v2 is not.

## Setup

- Same architecture as v1 (tiny byte LSTM).
- Pretrain: stripped Tiny Shakespeare **only**. No NOTE batches, no copy loss.
- Retrieve: longest stored snippet that starts with the probe, prepended as raw bytes. After love: context is `my love\nmy lo`. The string `NOTE:` does not appear.
- Same controls: disable-S, ρ reset, S reset, weight hash, twins.

## Headline (8× `my love`, probe `my lo`)

| Check | v1 NOTE-copy (S on) | v2 raw retrieve (S on) | v2 S off |
|-------|---------------------|------------------------|----------|
| NOTE in probe context | yes | **no** (`my love\nmy lo`) | no |
| Weights unchanged | yes | yes | yes |
| Inspectable fact in S | yes | yes | no |
| Empty prior P(v) | 0.027 | 0.084 | 0.084 |
| P(v) before ρ reset | 0.999 | 0.555 | 0.528 |
| P(v) after ρ reset | **0.988** | **0.093** | **0.084** |
| ΔP(v) vs prior after reset | +0.96 | **+0.009** | +0.000 |
| Reset S | → prior | → prior | n/a |

Classification: **Trace-only**. Session residue moves P(v) (~0.53 before reset). ρ reset wipes it. Raw S is sitting in the prompt and the tiny LM **does not** use it.

## What this means

v1’s Store-works was real but **protocol-dependent**: the species prior had been taught to copy `NOTE: pfx -> ch`. Take that lesson away, keep the inspectable store and RAG-style replay, and this model falls back to BDH Category B.

That is not a reason to sneak NOTE training back in. It is the result.

## Honest limits

- A larger LM with in-context learning might use `my love\nmy lo` without a special format. **Not measured.**
- Snippet replay still puts the letter `v` in the window (`my love`). The LSTM still did not prefer `v` at the probe. So this is not “we hid the answer.”
- Do not claim v2 refutes stores in general. It refutes **this** frozen tiny LSTM using **untaught** raw retrieve.

## Reproduce

```bash
python -m experiments.train_prior --plain
python -m experiments.run_v2
```
