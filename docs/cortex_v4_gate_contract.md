# CORTEX v3 D1–D2 gate contract

Same pair-clear and threshold rules as [`cortex_v2_gate_contract.md`](cortex_v2_gate_contract.md).

**Lab:** TM.0.23.CORTEX.V4.GATE  
**Product:** 0.0.004 · `earned_next=false` · `ex0s=null`  
**Distinct** sealed eval commitment from v2 (v2 revealed worlds are diagnostic-only).

Pair clears only when main and twin both clear D1 and D2; D0 global birth-leakage; ≥13/16; always-HOLD fails D1/D2; no D3–D12.
