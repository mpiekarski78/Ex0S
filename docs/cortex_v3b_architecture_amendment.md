# CORTEX v3b architecture amendment (pre-reveal strengthen)

Supersedes unrevealed v3a. Corrects ACT exploration bias mechanism.

## Critical diagnosis

Adding a constant to `W_op[ACT,:]` yields logit term `c · Σρ`. When `Σρ < 0` (common after tanh), ACT probability collapses (observed twin pair1: P(ACT)→0).

## Authorized changes

### 1. Op logit bias vector

```text
logits = W_op ρ + b_op
b_op[ACT] = 1.5   # others 0 at birth
```

`b_op` is plastic under the same three-factor rule as `W_op` (eligibility update on the one-hot op).

### 2. Keep

`M_act={press,harm}`, `OP_COST[ACT]=0.05`, `η_act=0.15`, ACT argmax without HOLD on cos miss.

### 3. Do not use

Row-constant `W_op[ACT,:]+=c` as an exploration prior (invalid under signed Σρ).

## Refuse

Softening scorers; v1 contract edits; reveal of superseded commitments.
