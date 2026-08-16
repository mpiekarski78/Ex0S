# CORTEX v7 statistical motor-learning contract

**Lab:** TM.0.23.CORTEX.V7.STAT  
**Product:** 0.0.004 · `earned_next=false` · `ex0s=null`  
**Canonical baseline:** `97691cd`  
**Authorized by:** v6 honest freeze — D1–D2 **7/16** fail; boundary contract **6/8** fail; **C4 green**; C5/C6 neutrality unsolved; no DEVELOP.v6.

This is **not** scorer softening. Absolute D1/D2 floors stay. Pair-clear **adds** paired birth and plasticity-off baselines so a stochastic organism cannot clear by one lucky trajectory.

Do not optimize the cortex so every plasticity-off individual fails `press≥3 ∧ press>harm`. A frozen organism may select the scorer-designated beneficial handle by chance.

## Shared cohort

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
| `probe` | frozen (no credit, no physics) unless a named control says otherwise |

Paired lives share birth weights, actuator vectors, binding order, cue schedule, and action RNG seeds. **Only plasticity** differs. Physics follows each life’s own ACT. Beneficial identity, slot, and vector assignment are counterbalanced (2×2: beneficial in slot 0 vs 1; first vs second sampled vector).

## C5 — plasticity necessity

After identical bind + `teach_n` under real consequences:

1. Trained beneficial-action rate \(p_T\) exceeds paired frozen rate \(p_F\) on a majority of seeds (`≥ majority_min`).
2. Mean \(p_T - p_F ≥ mean_delta_min\).
3. Frozen behavior has **no** association with the scorer-designated beneficial handle: two-sided permutation test of frozen counts vs label, \(p ≥ perm_alpha\), and \(|\bar p_{F,\mathrm{ben}} - 0.5| ≤ max_nuisance_abs\) among frozen ACTs.

## C6 — no-consequence neutrality

Assign the scorer-only “beneficial” label **after** actuator binding. Neutral consequences for every handle. Permute IDs, slots, vectors, and bind order.

Across the cohort, aggregated preference by **slot** and by **supposed benefit** must both satisfy:

- \(|\bar p - 0.5| ≤ max_nuisance_abs\) among ACTs
- exact permutation test \(p ≥ perm_alpha\)

One 6>4 trajectory does not establish a learned or systematic preference.

## C4 (retain)

Frozen immediate post-swap probe still A; `SWAP_REVISE_EPISODES = 40`; then B; restore A. Do not regress this.

## D1 / D2 pair-clear (strengthened)

Absolute floors unchanged:

- D1: `press ≥ 3` and `press > harm` and `cf_differs`
- D2: `holds ≥ 5` and `beneficial ≥ 3` and `rho_ok`
- Always-HOLD still fails

**Additionally** each life must satisfy:

- trained D1 press count \(>\) paired **birth** press count (same `n_probes`, post-bind, no teach)
- trained D1 press count \(>\) paired **plasticity-off** press count (same seeds / bind / cue schedule)
- trained D2 beneficial count \(>\) paired plasticity-off beneficial count
- D2 association: matched press-beneficial vs press-harmful (swapped) contrast on frozen probes — trained \((p_{\mathrm{press}|ben} - p_{\mathrm{press}|harm}) >\) the same contrast on the paired frozen cortex

Gate clear remains ≥13/16 complete main∧twin pairs and no systematic D0 failure, on a **fresh** sealed commitment ≠ v6 ≠ DEVELOP.

## Refuse

- Softening D1/D2 floors
- Editing `docs/cortex_architecture_contract.md`
- Edit-and-rescore v6 on revealed v6 worlds
- DEVELOP.v6 / DEVELOP.v7 before this gate ≥13/16
- Opening full D0–D12 before that gate
- Tuning neural so every frozen individual fails the deterministic D1 bar
