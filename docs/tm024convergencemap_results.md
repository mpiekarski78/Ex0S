# TM.0.24.CONVERGENCEMAP DEV

Decision: **oracle_separability_not_operationally_reachable**.

Live robust: all C0/C1/C2/C3 false. Replay robust: C1/C3/C4 false.

P1 is still D1-separable (TRACEBRIDGE B4). Revisiting those examples under the frozen sequential constraints does not make an 8-cue map operational. Write-geometry closed. SCORE unopened. No neural candidate. C4 remains ceiling-only. 512/1536 budgets stay closed. Product **0.0.004**. `earned_next=false`.

## Coverage

220 cells, 220 unique IDs, `exposure_mode` live 176 / replay 44. Manifest SHA `7ad1feb895e6c28897be7dbafe082f5e8a50acbeb0f64695f18f2aea88d9fb7e`. Unique IDs and manifest were asserted before writing DEV. C1/C3 at k=16 live vs replay IDs are disjoint. PA infeasible updates: 0. Default `act_score_mode=query`. W1 not resurrected.

`k` is complete passes through the cue set. Live P1 is regenerated every exposure. Replay uses only frozen first-pass P1 rows. Retention probes do not update learner weights or organism state.

## What was measured

| Arm | Mode | k | 8-cue rank | 4-cue rank | 2-cue rank / twin | Eco / REST |
| --- | --- | --- | --- | --- | --- | --- |
| C0 always-update D3 | live | 1 | 0/4 | 0/4 | **4/4 / 2/2** | 0 / **1** |
| C1 error-only | live | 1 | 0/4 | 0/4 | **4/4 / 2/2** | 0 / **1** |
| C1 | live | 2–8 | 0/4 | 0/4 | **4/4 / 2/2** | 0 / **1** |
| C1 | live | 16 | 0/4 | 0/4 | 0/4 / 0/2 | 0 / 0 |
| C1 | replay | 16 | 0/4* | 0/4 | **4/4 / 2/2** | — |
| C2 PA one-pass | live | 1 | 0/4 | 0/4 | 2/4 / 0/2 | 0 / 0 |
| C3 PA | live | 2–16 | 0/4 | 0/4 | 0/4 / 0/2 | 0 / 0 |
| C3 | replay | 16 | 0/4 | 0/4 | 0/4 / 1/2 | — |
| C4 sequential RLS | replay | 16 | **4/4** | **4/4** | **4/4 / 2/2** | 0 / **1** |

\*C1 replay 8-cue: final live ranking is 4/4 and train ranking is 4/4, but `retention_ok` is false, so the cell fails.

C0 vs C1 at k=1 match on rank/twin/REST and both fail eco: they differ only by always-update vs error-only, as frozen.

Live k=16 on C1 loses even the 2-cue map that k=1..8 held. Extra live history is not free.

C4 ranks 2/4/8 cues, twins, and REST on stored P1 plus live probes. Ecological reversal fails, so C4 is not robust. It stays diagnostic ceiling-only. `implementation_authorized=false`.

## Ladder (disjoint, frozen order)

1. Compact error-correcting write: **false** (C2 and C1@k=1 are not robust).
2. Repeated live exposure suffices: **false**.
3. Exact replay of C1/C3 with live C1/C3 failing: **false** (replay C1/C3 are not robust).
4. Only C4: **false** (C4 eco fails).
5. First-match: **oracle_separability_not_operationally_reachable**.

Eight-cue oracle separability is not operationally reachable under these sequential constraints: one-shot D3, error-only perceptron with live repetition, passive-aggressive margin correction, exact replay of those compact rules, or sequential RLS replay with the required ecological cell.

## What this does not authorize

Do not install a 512-row event-end trace. Do not change the v29 write law. Do not resurrect W1. Do not declare v31/v32. Do not open SCORE. Do not treat C4 as an implementation. A later freeze would need a different operational class than the compact local rules tested here.
