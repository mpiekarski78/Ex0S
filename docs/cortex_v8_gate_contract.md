# CORTEX v8 D1–D2 narrow gate contract

**Lab:** TM.0.23.CORTEX.V8.GATE  
**Product:** 0.0.004 · `earned_next=false` · `ex0s=null`  
**Stat contract:** [`cortex_v8_stat_contract.md`](cortex_v8_stat_contract.md)

Same pair-clear cardinality as v2–v7 (≥13/16). Floors unchanged. Extras use birth-weight pairing, frozen probes, and `life_delta_min=0.10`. Lives are derived from the sealed eval seed after reveal.

```text
sensorimotor_association_gate_clear ⇔
  (≥13/16 complete main∧twin D1∧D2 pairs clear under the v8 stat contract)
  ∧ (no systematic D0 birth-leakage failure)
```

Always-HOLD fails. No D3–D12. Failed revealed worlds are diagnostic-only. Distinct sealed commitment from v2–v7 and DEVELOP.
