# CORTEX v5 D1–D2 narrow gate contract

Same pair-clear and threshold rules as [`cortex_v2_gate_contract.md`](cortex_v2_gate_contract.md).

**Lab:** TM.0.23.CORTEX.V5.GATE  
**Product:** 0.0.004 · `earned_next=false` · `ex0s=null`  
**Distinct** sealed eval commitment from v2–v4 and from DEVELOP.v5.

## Thresholds (frozen; do not soften)

| Stage | Pass rule |
|-------|-----------|
| D0 | Birth leakage gate from `score_d0` |
| D1 | beneficial-handle count ≥ 3 and > harm-handle and `cf_differs` |
| D2 | `rho_ok` and beneficial ≥ 3 and holds ≥ 5 |
| Always-HOLD | Fails D1 and D2 |

Handles are **opaque** environment IDs bound via `bind_actuators`; scorers count by role keys in `curriculum_tokens` (same numeric thresholds as v4).

## Gate clear

```text
sensorimotor_association_gate_clear ⇔
  (≥13/16 complete main∧twin D1∧D2 pairs clear)
  ∧ (no systematic D0 birth-leakage failure)
```

## Prerequisites

All seven (+shuffled-credit) M_act boundary controls green on candidate v5 before reveal/scoring.

## Isolation

Failed candidate cannot be edited and rescored on revealed v5 worlds. Full D0–D12 requires **DEVELOP.v5** on a later fresh commitment.
