# TM.0.30.KEYGEOM results

Frozen key-geometry diagnostic wall on v36 (`separated_key` training). Product **0.0.004**.

## Outcome vector (descriptive)

- `key_rho_history_drift`: **not_observed**
- `key_rho_perturbation_drift`: **not_observed**
- `separator_added_collisions`: **observed**
- `novel_overlap_false_familiarity`: **observed**
- `geometry_wall_complete`: **True**

## Distribution highlights

- Raw hist identity drift rate: 0.0
- Raw pert identity drift rate: 0.0
- Sparse hist identity drift rate: 0.0
- Sparse pert identity drift rate: ~0.016
- Novel false-familiarity rate: 1.0 (64/64 novel probes)
- Raw tie rate: 0.0
- Sparse tie rate: ~0.10
- Baseline raw≠sparse slot mismatches: 0/64
- Separator-added on perturb when raw stable: 18 trials

## Interpretation (descriptive; not v37 routing)

Early `key_rho` geometry is **stable** under history and σ=0.01 perturbation on these checkpoints. Identity drift and false familiarity enter through the **Hadamard separator / sparse overlap path**, not through raw nearest-`key_rho` retrieval. This is consistent with TM029: `early_raw` passed taught-stable probes while `separated_key` hist/pert and novelty failed.

## Wording

v36 killed the **event-end P1 retrieval** story (`raw_p1` reinstatement wall on TM029). It did **not** kill raw retrieval generally — `early_raw` passed ordinary taught-stable probes.

No v37 architecture, candidate lock, or threshold change in this pass.

## Post-DEV comparison (closure; not a decision-lock amendment)

Computed from the frozen DEV lock. No rerun. No threshold written into `lineage_keygeom.decision.lock`.

Definitions:

- **W** = max raw nearest-stored `key_rho` L2 on taught cues under history and perturbation
- **N** = min raw nearest-stored `key_rho` L2 on novel probes
- **B** = min pairwise L2 among stored taught `key_rho` vectors

| | all | core c8 | 8×4 |
|---|---|---|---|
| W | 0.294 | 0.294 | 0.288 |
| N | 0.603 | 0.603 | 0.603 |
| B | 0.688 | 0.688 | 0.688 |
| W < N | yes | yes | yes |

**Route:** `W < N` and W does not approach B (W/B ≈ 0.43). Raw space already contains a familiarity gap. v37 should test early raw indexing with a **preregistered** raw-distance familiarity rule on fresh domains. Do not jump first to a learned separator: TM030 shows raw identity was already correct and the fixed Hadamard + k-WTA separator damaged it.

**N caveat:** all 64 novel probes share one raw distance. Each probe is a checkpoint clone; the first post-checkpoint unused spelling draws the same `rng_registry` embedding regardless of token string. N is that unused-token distance, not 64 independent novel geometries. World/scale cells also share the same 8 stored keys (vocab allocated by first-sight order). The W < N gap is still strict (0/64 novels ≤ W).

## Separator collisions: 18 / 1280

| family | collisions (raw stable, sparse drifted) | perturbation trials |
|---|---|---|
| core c8 | 10 | 640 |
| 8×4 | 8 | 640 |
| **all** | **18** | **1280** |

Rate 18/1280 = 1.4% of all trials (18/1138 ≈ 1.6% excluding 142 sparse integer-overlap ties). Occasional separator brittleness, not broad identity collapse. Raw identity drift: **0/1280**.

Sparse separator remains the failed v36 control. Any raw familiarity rule belongs to a separately frozen v37 battery. Product **0.0.004**.

