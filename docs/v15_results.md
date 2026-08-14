# v15 results: joint write / schema / use / pick

**Date:** 14 August 2026  
**Classification:** **Store-works**  
**Run:** `runs/2026-08-14_093521_v15`

## Question

> Do the v13–v14 boxes still work when **none of them is clamped on**?

v14 A/B passed with `force_use` / `force_write`. v15: unique files + write WHEN + schema + use-gate + pick-one, all learned. Train on **red** free lives with a stale wrong note in S. Held-out green (stale `open` already in S). Cortex frozen. Generic copy. W has no answers. `n_forced=0`. Probe greedy. No door/`action=` in any head.

## Predeclared

| ID | If |
|----|----|
| Confound | Cortex hash moves; disable-S still `use_key`; answer in W; `n_forced > 0`; probe explores; a head sees door id |
| Fail | Untrained already `use_key`; red works, green doesn’t; empty S still `use_key`; apply-all still works; a gate had to be clamped to rescue the plot |
| Store-works | Cortex frozen; all four heads changed; untrained not `use_key`; trained red `use_key` (newest complete note); held-out green `wait`; empty S / disable-S `open`; apply-all still mixes |

Do not restore the USE_KEY/WAIT table. Do not put the motor act in a head. Do not re-clamp `force_use` / `force_write`.

## Headline

| Check | Result |
|-------|--------|
| Untrained conflict (stale `wait` + new `use_key`) | **`open`** (use-gate off) |
| Trained red after ρ reset | **`use_key`** (`use=True`, `one=True`; files `d0_stale` + `d0_t12_1` `action=2`) |
| Held-out green | **`wait`** (`d2_stale` `action=1` + `d2_t1_1` `action=0`) |
| Apply-all red (pick off) | `wait` |
| Empty S / disable-S | `open` |
| Heads changed | write, schema, use, pick |
| Cortex SHA256 | unchanged |
| Clamps | none |
| Train last 50 | 0.74 |

## What this means

v14: each box works **if the others are held open**.  
v15: they compose under split credit. Noisier than isolated A (0.90) / B (0.78), but the probe still reads the newest complete file.

## Honest limits

- Newest-wins, two write templates, and `logits[int(action)] += 3.0` are still genome.
- Credit is **split** (write/schema vs use/pick), not one shared return. Shared return would likely starve the write head.
- Untrained conflict is `open` because the use-gate starts off, not because apply-all mixed. Mix is the apply-all **control** after training.
- Tiny linear gates. Not a general learner.

## Reproduce

```bash
python tests/test_v15.py
python tests/test_v14.py
python -m experiments.run_v15
```
