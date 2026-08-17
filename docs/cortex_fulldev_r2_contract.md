# CORTEX FULLDEV.R2 — full D0–D12 after the D3.R3 isolated clear

**Lab:** TM.0.23.CORTEX.FULLDEV.R2  
**Product:** 0.0.004 · `earned_next=false` · `ex0s=null`  
**Authorized by:** [`cortex_d3_r3_gate.clear.note.lock`](cortex_d3_r3_gate.clear.note.lock)

One continuously developing `make_cortex` life per seed, live candidate v16. No capability flags. D0–D12 scorers remain [`cortex_develop_scorers.py`](../experiments/cortex_develop_scorers.py). Do not edit those scorers. Do not edit neural during this recorded run.

Worlds are derived from a new sealed 256-bit commitment with domain `TM023.FULL.R2.`. That domain is unused by FULLDEV.R1, D3.R1–R3, TM023.V8–V16, and the historical `pair_seeds` table.

Refuse:

- D0–D12 on revealed FULLDEV.R1, D3.R*, or V8–V16 worlds
- reuse DEVELOP.v1–v4 commitments or `pair_seeds()`
- stamp `earned_next` / product 0.0.005 from this run
- edit-and-rescore after seeing results
- neural edits during the recorded run
- open nursery before this battery clears

Gate: ≥13/16 pairs clear required D stages. `eligible_for_000005` stays false until a later nursery conversation actually clears.
