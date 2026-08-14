# v13 results: copy `action=` from the file

**Date:** 14 August 2026  
**Classification:** **Store-works**  
**Run:** `runs/2026-08-14_092302_v13`

## Question

> Can a boxed gate learn **to copy the integer in the file into the motor**, without a frozen USE_KEY/WAIT table, and without door identity in the gate?

v7–v12 applied a hardcoded `if action==2: use_key elif action==0: wait` table. A 4×4 map trained only on red would not transfer to green. v13: untrained gate **off** → ignore the tag (species prior `open`). When the gate is on: frozen generic `logits[int(action)] += 3.0` only. Train write+use on **red** free lives. Held-out: green `wait`, blue `open`. Retrieve stays frozen **select**. W has no `d0`/`d1`/`d2`. `n_forced=0`. Probe greedy.

## Predeclared

| ID | If |
|----|----|
| Confound | Cortex hash moves; disable-S still `use_key`; answer in W; `n_forced > 0`; probe explores; use-head gets door id |
| Fail | Use-head unchanged; planted `d0.tag` already `use_key`; red works but green/blue don’t; empty S still `use_key` |
| Store-works | Cortex frozen; use-head changed; untrained (empty or planted) `open`; trained red `use_key`; held-out green `wait` and blue `open`; empty S / disable-S `open`; dump-all still mixes |

Do not put door id in the use-head. Do not emit `use_key` with no file. Do not restore the if/elif table on the v13 path to rescue the plot.

## Headline

| Check | Result |
|-------|--------|
| Untrained empty S | `open` |
| Untrained planted `d0.tag` | **`open`** (gate off) |
| Trained red after ρ reset | **`use_key`** (`use=True`; file `action=2`) |
| Held-out green | **`wait`** (file `action=0`; never in use-head training) |
| Held-out blue | **`open`** (file `action=1`) |
| Dump-all red (`d0`+`d2`) | `wait` |
| Empty S / disable-S | `open` |
| Cortex SHA256 | unchanged |
| Use head | changed |
| Train last 50 | 0.84 |

Green is the transfer test: prior prefers `open`; the integer `0` is only in `d2.tag`. Blue `open` matches both the prior and `action=1`, so it is a weaker check.

## What this means

v12: learn **not to dump**.  
v13: learn **to read the file’s integer**, not a wired USE_KEY/WAIT map.

The fact is still in S. The genome still does not store it. The new skill is a gate over `{s_hit}` plus a generic copy.

## Honest limits

- Generic copy is still a frozen `logits[act] += 3.0`. The head does not invent a new motor act.
- Use features are `{s_hit, 0}`. High bias after training means empty-S still sets `use=True`, but there is nothing to copy, so the prior `open` remains.
- v7–v12 keep the old table (`use_read=False`). This path is opt-in.
- Tiny boxed linear gate. Not a general tool-user.

## Reproduce

```bash
python tests/test_v13.py
python tests/test_v12.py
python -m experiments.run_v13
```
