# TM.0.32.RESTSPLIT contract

**Lab:** TM.0.32.RESTSPLIT · **Organism:** v37 (frozen; no neural edit)

Fresh `TM032.RESTSPLIT.DEV./TWIN.` worlds. Standalone runner. Do **not** patch TM031 or add `early_raw_half_spacing` to `ACT_RECALL_MODES`.

## Purpose

TM031 case 3: retrieval identity and half-spacing familiarity held; eight-cue acquire misses were pre-REST stored-P1 actuator nonconvergence; REST later repaired the same mapping. Exhausted awake budget does **not** prove that more awake rehearsal is the mechanism. This battery splits one post-awake checkpoint across matched clones.

## Arms (one snapshot, five clones)

| Arm | Intervention |
|-----|----------------|
| `none` | No additional processing |
| `awake_only` | Gated violation-driven rehearsal only (`mix_slow=False`, no REST wrapper). Pass budget `EPISODE_REPLAY_EPOCHS` with early stop at zero store violations; stop early if cumulative updates reach the sibling `full_rest` `n_replay` (**compute-matched** to REST replay) |
| `replay_no_mix` | REST idle ticks + gated replay, **no** slow mix, then REST cleanup (`reset_rho`, grow/prune, `dev_epoch`) |
| `mix_only` | One `W_act_query` slow mix; no gated updates; no REST wrapper |
| `full_rest` | Unchanged `rest_epoch(n_rest_ticks)` |

No key, radius, separator, episode-lifecycle, or familiarity changes.

## Records

For every arm: store-row violations; retrieved-P1 actuator ranking and geometric margin; live cue ranking/margin; updates used and first-converged pass.

**Fixes** (routing gate): `n_store_violations==0` and live ranking 8/8. Margin is recorded, not a gate.

## v38 routing (diagnostic splits only)

Splits whose `none` arm already fixes are `baseline_already_converged` and do not route v38.

| If | Then v38 |
|----|----------|
| `awake_only` fixes | adaptive violation-driven rehearsal |
| else `mix_only` fixes | on-demand consolidation |
| else `replay_no_mix` fixes | REST-replay without mix |
| else `full_rest` fixes | credit-triggered micro-REST |
| else nothing compute-matched fixes | optimization/capacity wall |

Battery first-match requires diagnostic splits to agree; mixed routes stay mixed. Product **0.0.004**.
