# CORTEX v2 architecture amendment — TM.0.23.CORTEX

**Lab:** TM.0.23.CORTEX.DIAG → candidate v2  
**Product under test:** Ex0S **0.0.004**  
**Flags:** `earned_next=false`, `ex0s=null`  
**Does not edit:** [`cortex_architecture_contract.md`](cortex_architecture_contract.md) (v1 remains pinned)

## Authorization

Only changes listed below are permitted. Each cites [`cortex_diagnosis.lock`](cortex_diagnosis.lock).

## Equations / constants changed

### 1. Birth-time motor lexicon

**v1:** ACT/EMIT tokens chosen by cosine match over the full sensory `vocab` grown from observe symbols. Motor operands `press`/`harm`/`get`/`drop` are not pre-registered.

**v2:** At birth, register dedicated motor lexicon

```text
M_act = {press, harm, get, drop, idle}
```

with registry-seeded unit-scale vectors (`seed_registry`). ACT token selection searches **only** `M_act`, never the sensory vocab.

**Evidence:** diagnosis rank-1 `motor_lexicon_absent` — `vocab_ever_has_press=false`; ACT tokens were `st_idle`.

### 2. ACT cosine miss no longer forces HOLD

**v1:** `_best_token`: if `max cos < cos_thresh` → `None` → motor loop forces HOLD.

**v2:** For `op=ACT` only: select `argmax_{t in M_act} cos(W_act_query ρ, v_t)` even when that cosine is below `cos_thresh`. EMIT retains v1 behavior (full vocab + cos_thresh → HOLD).

**Evidence:** diagnosis rank-2 `forced_hold_cosine` — `forced_hold_suspect_rate≈0.71`, `hold_rate≈0.79`.

### 3. ACT operation cost

**v1:** `OP_COST[ACT] = 1.0`

**v2:** `OP_COST[ACT] = 0.1`

Advantage remains:

```text
adv = ‖body_prev − body*‖ − ‖body_t − body*‖ − cost(op)
```

Under DEFAULT_LATENT teach physics, beneficial `press` from BODY0 must yield `adv>0`; `harm` must yield `adv<0`.

**Evidence:** diagnosis rank-3 `act_cost_dominates_advantage` — credited ACT mean `adv=-1.0`; `delta_p_act<0`.

## Inherited unchanged from v1

- Observe ABI exact keys / banned keys  
- Ops set RETRIEVE/WRITE/EMIT/ACT/STOP/HOLD  
- Genome sizes `n,d_sym,k_s,d_body,p_connect,T_max,τ,cos_thresh` (EMIT path), `η_pred,η_act,β,clip`  
- Sequential sensory microticks; directed prediction; three-factor credit structure  
- Retrieval buffer timing; WRITE commit after motor loop  
- ρ reset clears activation/buffer/eligibility without weight change  
- No wrap of `make_interpret` / `ThreeMemoryAgent`  
- dtype float64 CPU gold; body setpoint `[1,0,1,0]`  
- Costs for RETRIEVE/WRITE/EMIT/STOP/HOLD unchanged  

## Explicit refusal of unrelated improvements

No larger cortex, new ops, attention redesign, reward keys, scorer softening, editing the v1 architecture contract, or full D0–D12 on gate worlds.
