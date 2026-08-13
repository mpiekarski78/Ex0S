# v8 results: boxed use-policy (cortex frozen)

**Date:** 14 August 2026  
**Classification:** **Store-works**  
**Run:** `runs/2026-08-13_235752_v8`

## Question

> Can a tiny boxed policy learn *when* to commit a matched tag and *when* to apply it, while the frozen cortex never stores the fact?

v7 proved native tags work with a **hardcoded** collect/apply rule. v8 lets that box change. Cortex SHA256 must not. The file’s `action=` still chooses the motor act via frozen `_apply_record_bias`. The policy must not learn “red → use_key”.

## Predeclared

| ID | If |
|----|----|
| Confound | Cortex hash moves, **or** disable-S after ρ reset still `use_key` (fact leaked) |
| Fail | Policy hash unchanged; red unmount wrong; held-out green wrong (learned *that door*) |
| Store-works | Cortex unchanged, policy changed, red commit+unmount `use_key`, empty S → `open`, held-out green works without retraining cortex |

Held-out: green door (`door=2`), file `d2.tag` `{door:2, action:0}`. Probe `probe_green`: **WAIT** opens (prior prefers OPEN). Never in training W.

Policy features: **`s_hit`, `w_hit` only** (n_feat=2). No door id. No novelty (novelty would leak the observation).

## Headline

Untrained apply gate starts off (`b_apply = -1.2`). Empty prior at red: **`open`**.

| Check | Action | Correct? |
|-------|--------|----------|
| Untrained red (W on, apply off) | open | no (wanted) |
| Red: commit from W, then unmount W, ρ empty | **use_key** | **yes** |
| Empty S after delete | open | no |
| disable-S, W unmounted, ρ empty | open | no |
| Held-out green, unmount W, ρ empty | **wait** | **yes** |
| Cortex SHA256 | unchanged | yes |
| Policy SHA256 | changed | yes |
| Train return last 50 | 0.92 | — |

S after red: only `d0.tag`. S after green: only `d2.tag`. Action on green is `wait` because that is what the **file** says, not because the policy memorized a door.

## What trained

Each episode is two probes on red:

1. W mounted, S empty → features `[0, 1]` → learn to **commit**.
2. Unmount W, reset ρ → features `[1, 0]` → learn to **apply** from S.

Reward is only on step 2. After training, `[0, 1]` still has low `p_apply` (~0.06): the first glance commits and does not yet use the tag. After unmount, `[1, 0]` has `p_apply` ≈ 0.999 and `use_key` / `wait` come from the committed file.

disable-S is probed with **W unmounted**. Leaving W mounted would let a peek apply the tag without S (false Confound).

## What this means

The boxed skill is **when to take and when to use a matched record**. World knowledge stays in `.tag` files. Cortex stays the genome (seed 1337). Held-out green is the check that the policy did not learn “that door.”

v7 was the same split with a frozen collect rule. v8 is the first time the *use* box is allowed to move.

## Honest limits

- Linear REINFORCE, two features, 400 red episodes. Not a general learner.
- Motor act is still the frozen tag→logit grammar, not a policy output.
- First step with W on commits and still `open`; use happens after unmount. That is the trained feature split, not “instant read of W.”
- Peek control: the trained policy **commits**, so the empty-S check deletes files rather than relying on a peek-only arm.

## Reproduce

```bash
python tests/test_v8.py
python tests/test_v7.py
python -m experiments.run_v8
```
