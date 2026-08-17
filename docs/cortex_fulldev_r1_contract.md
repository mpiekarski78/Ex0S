# CORTEX FULLDEV.R1 — full D0–D12 after the v13 narrow clear

**Lab:** TM.0.23.CORTEX.FULLDEV.R1  
**Product:** 0.0.004 · `earned_next=false` · `ex0s=null`  
**Authorized by:** [`cortex_v13_gate.clear.note.lock`](cortex_v13_gate.clear.note.lock)

One continuously developing `make_cortex` life per seed. No capability flags. D0–D12 scorers remain [`cortex_develop_scorers.py`](../experiments/cortex_develop_scorers.py) (audited R1+R2). Do not edit those scorers in this cycle.

Worlds are derived from a new sealed 256-bit commitment with domain `TM023.FULL.R1.`. That domain is unused by TM023.V8–V13 gates and is not the historical `pair_seeds` table used by DEVELOP.v1–v4 and C4/C5/C6.

Refuse:

- D0–D12 on revealed v13 (or any V8–V13) gate worlds
- reuse DEVELOP.v1–v4 commitments or `pair_seeds()`
- stamp `earned_next` / product 0.0.005 from this run
- edit-and-rescore after seeing results
- neural edits during the recorded run

Gate: ≥13/16 pairs clear required D stages. `eligible_for_000005` additionally needs ≥14/16 maturation; this pass still does not stamp.
