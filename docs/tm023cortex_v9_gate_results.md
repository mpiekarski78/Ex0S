# TM.0.23.CORTEX v9 gate

**sensorimotor_association_gate_clear:** `False`  
**n_pair_clear:** `6/16`  
**candidate git:** `bd94241`  
**device:** CUDA / NVIDIA GB10  
**D1 bind:** press+harm only  

| Stage | ok / 32 | floors / 32 |
|-------|---------|-------------|
| D0 | 32 | 32 |
| D1 | 23 | 32 |
| D2 | 21 | 27 |

press=0 lives: **0**. Every D1 life has `press>=3` and `press>harm`. The nine D1 reds fail only the per-life `life_delta_min=0.10` extras against birth or plasticity-off. Five D2 lives fail `holds>=5`. Floor-only pairs would be 11/16 — still below 13. Historical 6/16 stands. **DEVELOP.v9 refused.**
