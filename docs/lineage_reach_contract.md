# Lineage REACH contract — TM.0.24.REACH

**Lab:** TM.0.24.REACH  
**Subtitle:** Developmental reachability after credit-path repair  
**Product under test:** Ex0S **0.0.004**  
**Flags:** `earned_next=false`, `ex0s=null`, `eligible_for_000005=false`  
**Not:** TM.0.25, a new lineage version, a capability earn, QUAL/EVAL, or a WALLMAP rescore.

Authorized by [`lineage_wallmap.decision.lock`](lineage_wallmap.decision.lock) after candidate [`cortex.candidate.v28.lock`](cortex.candidate.v28.lock). Frozen LINEAGE and WALLMAP results remain historical. n stays **64**. Do not move τ or δ.

## Package question

After the general credit-path repair, can **one** Arm D genotype learn bounded L0 on unused renamed worlds under ordinary teaching and credit?

## Learning gates (same as WALLMAP Q2; new worlds)

Pass on `TM024.REACH.DIAG.CHECK.` with the same genotype optimized only on `TM024.REACH.DIAG.FIT.`:

- birth mean `< τ=0.60`
- plasticity-off mean `< τ=0.60`
- adult mean `≥ 0.60` and cluster-bootstrap CI lower bound `≥ 0.60` (`n_boot=9999`, seed `20260817`)
- `G_k` with `δ_B=δ_P=0.05`
- ordinary credit only; no Q1 weight inheritance; wired Arm D scalars only
- ≥ 4 CHECK worlds; sibling births

A favorable birth is not reachability. No per-evaluation-world genotype. Do not reuse `TM024.WALLMAP.Q2.DIAG.*` as held-out.

## Credit-path precondition

Before interpreting CHECK, record the v28 zero-eligibility probe: unused plastic tensors must not move when `ρ_elig=0`.

If that probe fails, a CHECK fail does **not** independently diagnose maturation/replay.

## Order of freezes

1. This contract + prereg (commit/push before answers)
2. Runner + tests; ABI/synthetic smoke only
3. `docs/lineage_reach.runner.lock` on clean `origin/main`
4. Score once
5. Result lock

## Refuse

Scorer softening; L0-specific circuitry; QUAL/EVAL reveal; WALLMAP Q2 world reuse as held-out; rewriting LINEAGE/WALLMAP locks; claiming impossibility; `earned_next`; 0.0.005; moving `τ`/`δ`; increasing n; FULLDEV.R7.
