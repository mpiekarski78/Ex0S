# CORTEX v4 architecture amendment

After v3 gate failure (revealed worlds diagnostic-only).

## Change

1. `b_op[ACT] = 0.85` at birth (was 1.5)  
2. `b_op` is **frozen** — not in plastic set; credit updates `W_op` only  
3. Inherit v3b: `M_act={press,harm}`, `OP_COST[ACT]=0.05`, `η_act=0.15`, ACT argmax without cos HOLD  

**Evidence:** [`cortex_diagnosis.v3.lock`](cortex_diagnosis.v3.lock) — D2 HOLD starvation from plastic ACT bias.

## Refuse

Softening scorers; editing v1 contract; reusing v3 commitment.
