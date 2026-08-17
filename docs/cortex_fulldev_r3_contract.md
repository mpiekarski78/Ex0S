# CORTEX FULLDEV.R3 — full D0–D12 after isolated D3.R3 and D4.R2 clears

**Lab:** TM.0.23.CORTEX.FULLDEV.R3  
**Product:** 0.0.004 · `earned_next=false` · `ex0s=null`  
**Authorized by:** [`cortex_d4_r2_gate.clear.note.lock`](cortex_d4_r2_gate.clear.note.lock)

One continuously developing `make_cortex` life per seed, live candidate v18. No capability flags. D0–D12 scorers remain [`cortex_develop_scorers.py`](../experiments/cortex_develop_scorers.py). Do not edit those scorers. Do not edit neural during this recorded run.

Worlds are derived from a new sealed 256-bit commitment with domain `TM023.FULL.R3.`. That domain is unused by FULLDEV.R1–R2, D3.R*, D4.R*, TM023.V8–V18, and the historical `pair_seeds` table.

Refuse:

- D0–D12 on revealed FULLDEV.R1–R2, D3.R*, D4.R*, or V8–V18 worlds
- reuse DEVELOP.v1–v4 commitments or `pair_seeds()`
- stamp `earned_next` / product 0.0.005 from this run
- edit-and-rescore after seeing results
- neural edits during the recorded run
- open nursery before this battery clears

Gate: ≥13/16 pairs clear required D stages. `eligible_for_000005` stays false until a later nursery conversation actually clears.
