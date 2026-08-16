# M_act boundary contract — TM.0.23.CORTEX

**Lab:** TM.0.23.CORTEX.MACT.BOUNDARY  
**Product:** Ex0S **0.0.004** · `earned_next=false` · `ex0s=null`  
**Under test (v1 runner):** frozen [`cortex.candidate.v4.lock`](cortex.candidate.v4.lock) planted motor dictionary

## Claim

`M_act` is valid only if actuators are **environment-exposed opaque handles**, not meanings planted in the genome. Innate `b_op[ACT]` and generic low `OP_COST[ACT]` are defensible motor tendencies. A fixture-specific motor dictionary is not.

## Controls (all must be green on candidate v5 before narrow D1–D2 re-earn)

| ID | Pass rule |
|----|-----------|
| `C1_env_exposed_handles` | Cortex does not birth-register English / `press` / `harm` as innate lexicon |
| `C2_spellings_not_neural_input` | Handle string is an external registry key only; never a sensory/vocab embedding input |
| `C3_twin_independent_rename_vectorize` | Distinct opaque handle IDs; separate motor-registry RNG streams; independently sampled nonsemantic vectors |
| `C4_consequence_swap_timed` | See swap protocol below |
| `C5_plasticity_off` | Same bindings; S + observations available; freeze cortical plasticity **and** eligibility updates; retain innate ACT bias/cost; learning-dependent D1/D2 portions **fail** |
| `C6_no_consequence` | Actions still execute; body/state consequences neutral and counterbalanced; no handle gains systematic preference; D1/D2 fail despite `b_op` |
| `C7_distractor_motors` | Extra non-beneficial handles available; must not become preferred without beneficial consequences |
| `C8_shuffled_credit` | Beneficial consequence follows an unrelated `interaction_token`; must **not** reinforce the previous actuator |

## C4 — Consequence-swap timing (preregistered)

Preference must not flip immediately merely because physics changed.

1. Learn handle A as beneficial.
2. Confirm A preference (`pref_A`: A count ≥ 3 and A > B on 40 probes).
3. Swap physical consequences **without rebinding vectors**.
4. Confirm **initial stale preference permitted** (`stale_ok`: still prefer A on immediate 20 probes).
5. Supply exactly **`SWAP_REVISE_EPISODES = 40`** new consequence episodes under swapped physics.
6. Require preference to move to handle B (`pref_B`: B ≥ 3 and B > A on 40 probes).
7. Restore pre-swap cortex checkpoint → preference returns to A (`restore_A`).

## V1 runner note

The v1 boundary runner scores **frozen candidate v4** (planted `MOTOR_ACT_TOKENS`). Expected reds on C1–C3 (and likely C4/C7). Results are diagnostic; they do not authorize softening scorers.

## V5 path

After documenting v4 reds: freeze v5 actuator ABI + boundary runner.v2 + narrow-gate commitment **before** neural edits. V5 must clear all controls then re-earn D1–D2 (≥13/16) before DEVELOP.v5.
