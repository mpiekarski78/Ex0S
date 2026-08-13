# v1 results: language probes vs BDH

**Date:** 14 August 2026  
**Classification:** **Store-works**  
**Checkpoint:** `checkpoints/prior.pt` (gitignored; train with `python -m experiments.train_prior`)  
**NOTE-follow acc (padded window 64):** 1.0  
**Run:** `runs/2026-08-13_223617_v1`

## Question

> Can experience write language facts into S such that, after ρ reset, completions still reflect those facts — while disable-S recovers BDH Category B on the same probes?

## Species prior (frozen)

Tiny byte LSTM (`n_embd=64`, `n_hidden=128`, 1 layer) trained on Tiny Shakespeare **with lord/love/`my lo` stripped**, plus aligned `NOTE: pfx -> ch` copy examples. Probe facts are **not** in the pretrain. Weights frozen for the life. SHA256 unchanged.

Empty prior on `my lo`: P(`r`)=**0.0006**, P(`v`)=**0.027** (no Shakespeare “lord” bias). BDH’s empty prior P(`r`)=0.636 / P(`v`)=0.181 is a different species prior, not a bug in this table.

## Headline (8× `my love` / `my lord`, probe `my lo`)

| Check | BDH (published) | three-memory S **off** | three-memory S **on** |
|-------|-----------------|------------------------|------------------------|
| Weights unchanged | yes | yes | yes |
| Empty prior P(v) | 0.181 | 0.027 | 0.027 |
| Empty prior P(r) | 0.636 | 0.0006 | 0.0006 |
| P(v) after 8× love, **before** ρ reset | ~0.61 | 0.252 | 0.999 |
| P(v) after 8× love, **after** ρ reset | effect gone (JS→0) | **0.027** (= prior) | **0.988** |
| ΔP(v) vs prior after ρ reset | n/a | +0.000 | **+0.961** |
| P(r) after 8× lord, after ρ reset | does not raise vs prior | — | **0.999** |
| JS(love, lord) after ρ reset | 0 | 0 vs prior | **0.686** |
| 1 extra filler byte | can wipe P(v) | 0.252→0.252 (keyed session) | 0.988→0.988 (S holds fact) |
| Reset S | n/a | n/a | P(v)→0.027 |
| Inspectable fact | no | no | `NOTE: my lo -> v` in JSON |
| Twin ρ L2 | 0 | 0 | 0 |
| ρ restore | exact | P(v) match | P(v) match |

disable-S before reset: ΔP(v)=**+0.225** (session prefix→byte buffer). After reset: **+0**. That is the BDH Category B analogue.

## Honest limits

- Retrieve is **explicit NOTE context** the frozen LM was trained to copy, not a mystery in ρ.
- S records every surprising 5-gram; probe uses the **longest suffix match** (`my lo -> v`).
- Session ρ is a **prefix-keyed** last-byte buffer plus hidden EMA. One filler byte does **not** scramble other keys (unlike BDH’s mixed linear-attention ρ). Durability after **ρ reset** is the comparison that matters; it is entirely S.
- Do not say the tiny LSTM is a better language model than BDH. It is a different prior plus a store.

## Reproduce

```bash
python -m experiments.train_prior   # writes checkpoints/prior.pt
python -m experiments.run_v1
```

Uses `/opt/BDH_v1/input.txt` if present (read-only). Does not import or modify `bdh.py`.
