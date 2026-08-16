# TM.0.23.CORTEX.DEVELOP results

**product:** `0.0.004`
**development_gate_clear:** `False`
**eligible_for_000005:** `False`
**earned_next:** `False`
**ex0s:** `None`

Pairs clear: **0/16** · Maturation: **0/16**

Locks: [`cortex_sanity_spec.amendment.lock`](cortex_sanity_spec.amendment.lock) · [`cortex_development.runner.lock`](cortex_development.runner.lock) · [`cortex_eval_reveal.lock`](cortex_eval_reveal.lock) · [`cortex_development.prereg.lock`](cortex_development.prereg.lock) · [`cortex_development.lock`](cortex_development.lock) · [`cortex_wall.lock`](cortex_wall.lock)

## Stage pass counts (main+twin)

- `D0`: 32
- `D1`: 0
- `D2`: 0
- `D3`: 14
- `D4`: 0
- `D5`: 4
- `D6`: 5
- `D7`: 0
- `D8`: 0
- `D9`: 12
- `D10`: 0
- `D11`: 4
- `D12`: 0

## Diagnostic wall

first_fail_neural_wall: `W_persist` (need not pass; cannot negate development gate)

## Note

Neural organism unchanged vs candidate. Capacity/wall diagnostic only.

**Scorer audit R2:** prior results preserved as [`cortex_development.v1.lock`](cortex_development.v1.lock) / [`.v2.lock`](cortex_development.v2.lock) / [`.v3.lock`](cortex_development.v3.lock). Fixes in [`cortex_develop_scorers.py`](../experiments/cortex_develop_scorers.py); see [`cortex_scorer_audit.amendment.lock`](cortex_scorer_audit.amendment.lock) and [`cortex_scorer_audit.r2.amendment.lock`](cortex_scorer_audit.r2.amendment.lock). Same eval commitment; contiguous `last_stage_clear`. Soft stages closed: D9 retention floor, D11 cross-lexicon, D5 known/unknown margin, D12 non-vacuous skill forks.
