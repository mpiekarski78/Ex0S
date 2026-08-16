# CORTEX v8 statistical motor-learning contract

**Lab:** TM.0.23.CORTEX.V8.STAT  
**Product:** 0.0.004 · `earned_next=false` · `ex0s=null`  
**Authorized by:** [`cortex_diagnosis.v7.lock`](cortex_diagnosis.v7.lock) + [`cortex_v7_gate.audit.lock`](cortex_v7_gate.audit.lock)  
**Canonical main at diagnosis:** `b2fc247` / prior result `ec10c03`

This is **not** floor softening and **not** a v7 rescore. Historical 0/16 stands.

v7 already showed population C5/C6 learning and C4 revision. Its gate extras were still one-life strict `>` after a 30-episode teach, mixing frozen-probe birth with `score_d1` apply_event counts, on worlds that did not depend on the sealed seed.

## Shared cohort (C5/C6 retain)

| Quantity | Frozen value |
|----------|----------------|
| `n_pairs` | 32 |
| `n_probes` | 40 |
| `teach_n` | 80 |
| `majority_min` | 24 / 32 |
| `mean_delta_min` | 0.10 |
| `max_nuisance_abs` | 0.15 |
| `perm_n` | 9999 |
| `perm_alpha` | 0.05 |
| `probe` | frozen (no credit, no physics) for extras |
| `life_delta_min` | 0.10 |

Paired lives share birth weights, actuator vectors, binding order, cue schedule, and action RNG. **Only plasticity** differs. Physics follows each life’s own ACT. Clone the plasticity-off twin from the **post-bind birth checkpoint**, not after a partial teach.

## C4 / C5 / C6

Retain the v7 population C5/C6 bars and C4 swap timing (`SWAP_REVISE_EPISODES=40`, immediate probe before post-swap teach).

## D1 / D2 pair-clear

Absolute floors unchanged:

- D1: `press ≥ 3` and `press > harm` and `cf_differs` on the **same frozen-probe** measure as the extras
- D2: `holds ≥ 5` and `beneficial ≥ 3` and `rho_ok` (existing `score_d2` floors)
- Always-HOLD fails

**Additionally** each life, from the birth checkpoint, same frozen probe:

- \(p_T - p_{\mathrm{birth}} ≥ life\_delta\_min\)
- \(p_T - p_F ≥ life\_delta\_min\)
- D2 association: trained frozen-probe contrast \(>\) paired frozen contrast

Gate worlds are derived from the **revealed eval seed**, not `10000 + 97·pair_id`.

Gate clear remains ≥13/16 complete main∧twin pairs and no systematic D0 failure, on a fresh sealed commitment ≠ v7 ≠ DEVELOP.

## Neural

No new credit law. Retain v7 `skip_act_cost` when `body_adv≈0`. Do not tune so every frozen individual fails deterministic D1.

## Refuse

- Softening D1/D2 floors
- Editing `docs/cortex_architecture_contract.md`
- Edit-and-rescore v7
- DEVELOP.v7 / DEVELOP.v8 before this gate ≥13/16
- Opening full D0–D12 before that gate
