# CORTEX v7 D1–D2 narrow gate contract

**Lab:** TM.0.23.CORTEX.V7.GATE  
**Product:** 0.0.004 · `earned_next=false` · `ex0s=null`  
**Baseline:** `97691cd`  
**Stat contract:** [`cortex_v7_stat_contract.md`](cortex_v7_stat_contract.md)

Same pair-clear cardinality as [`cortex_v2_gate_contract.md`](cortex_v2_gate_contract.md). Absolute D1/D2 floors unchanged. Pair-clear **also** requires trained > paired birth and trained > paired plasticity-off, plus D2 consequence association, as frozen in the stat contract.

Authorized only after [`cortex_diagnosis.v6.lock`](cortex_diagnosis.v6.lock) and the v7 architecture amendment.

## Gate clear

```text
sensorimotor_association_gate_clear ⇔
  (≥13/16 complete main∧twin D1∧D2 pairs clear under the v7 stat contract)
  ∧ (no systematic D0 birth-leakage failure)
```

Always-HOLD fails D1/D2. No D3–D12. Failed revealed worlds are diagnostic-only. Distinct sealed commitment from v2–v6 and DEVELOP.
