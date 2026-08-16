# TM.0.23.CORTEX v11 gate

**sensorimotor_association_gate_clear:** `False`  
**n_pair_clear:** `6/16`  
**candidate git:** `0d17abb`  
**device:** CUDA / NVIDIA GB10  
**D2 conflict:** swapped press/harm physics  

| Check | Result |
|-------|--------|
| D0 | 32/32 |
| D1 floors | 32/32 |
| D1 population | 31/32 vs birth, 29/32 vs frozen — green |
| D2 floors | 22/32 |
| D2 population | 27/32 vs frozen, assoc 32/32 — green |

Ten D2 reds fail `holds>=5` under swapped physics (eight at holds=4). Historical 6/16 stands. **DEVELOP.v11 refused.**
