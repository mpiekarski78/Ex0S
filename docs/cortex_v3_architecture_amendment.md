# CORTEX v3 architecture amendment — TM.0.23.CORTEX

**Lab:** TM.0.23.CORTEX after v2 gate failure  
**Product:** Ex0S **0.0.004** · `earned_next=false` · `ex0s=null`  
**Does not edit:** [`cortex_architecture_contract.md`](cortex_architecture_contract.md) or v2 amendment history  

Inherits v2 changes (motor lexicon, ACT no HOLD on cos miss, reduced ACT cost) and adds only the following.

## Authorization

[`cortex_diagnosis.v2.lock`](cortex_diagnosis.v2.lock) after [`cortex_v2_gate.failure.lock`](cortex_v2_gate.failure.lock).

## Equations / constants changed (v3 only)

### 1. Restrict ACT motor targets

**v2:** `M_act = {press, harm, get, drop, idle}`  

**v3:** ACT selection lexicon

```text
M_act = {press, harm}
```

`get`/`drop`/`idle` may still exist as ordinary vocab symbols if observed; they are **not** ACT targets.

**Evidence:** diagnosis.v2 rank-1 `motor_operand_dilution`.

### 2. Birth ACT logit bias

**v2:** `W_op` rows i.i.d.  

**v3:** After init,

```text
W_op[ACT, :] ← W_op[ACT, :] + 0.35
```

(also stored in birth / slow snapshots).

**Evidence:** diagnosis.v2 rank-2 `act_rate_still_fragile`.

### 3. ACT cost

**v2:** `OP_COST[ACT] = 0.1`  

**v3:** `OP_COST[ACT] = 0.05`

**Evidence:** diagnosis.v2 ACT `adv` mean still negative / low frac_pos.

## Inherited unchanged

All v1 contract terms not amended in v2/v3; all v2 amendments except where superseded above; D1/D2 scorer thresholds; observe ABI; no interpret wrap.

## Refuse

Unrelated improvements; edit-rescore on v2 revealed worlds; softening scorers; editing the original architecture contract.
