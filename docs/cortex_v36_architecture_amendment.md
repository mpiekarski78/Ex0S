# CORTEX v36 architecture amendment

Authorized by [`lineage_complementary.v35.closure.lock`](lineage_complementary.v35.closure.lock). v35 isolation is **not** edited.

v35 nearest-P1 recall misrouted learned cues and treated novel cues as familiar. Raw event-end P1 is not a reliable retrieval key. That result is closed at `42ce89f`.

v36 tests the **hypothesis** that an action-owned earlier event key, sparsely separated, with a familiarity gate, can index stored P1 for cortical scoring. PHASEMAP showed earlier-phase robustness in a different setting; it does **not** establish that this snapshot plus sparse hashing solves history invariance.

## Dual path (required)

1. **Write/replacement unchanged:** eight content-addressed P1 episodes; L2 radius **0.05**; contradictory replacement. Do not match or replace on key overlap.
2. **Store keys alongside episodes:** each slot may carry `key` (binary 8-of-64) and `key_rho` (raw early snapshot). Missing legacy keys → cortical fallback.
3. **Recall is the only new path.** Never return stored handle.

## Key construction

- Capture rho after ordered-symbol ticks, **before** `v_end` (`_last_key_rho`). This earlier capture is a hypothesis, not a prior fact.
- Sparse key: deterministic **normalized 64-d Hadamard** with fixed sign/permutation; k-WTA of the **largest signed** activations; equal activations use **lower index**. Binary **8-of-64** (12.5% sparsity — unrelated to episode slot count).
- Separator budget: 4096 fixed coefficients. Pin the **complete matrix SHA**, not only a seed. Not biological DG expansion; a sparse hash.
- Action-owned: `pending["event_key"]` and `pending["key_rho"]` copied at ACT beside `rho_p1`. Delayed credit writes the **pending** key, never live `_last_event_key`.

## Familiarity

Integer overlap ≥ **5/8** to retrieve. Null bound for independent 8-of-64 keys: P(overlap ≥ 4) ≈ 0.00617/episode (~4.8% across 8 slots); P(overlap ≥ 5) ≈ 0.000361/episode (~0.29% across 8). 4/8 is too permissive. Exact integer ties at maximum overlap → cortical fallback. Organism keys may be correlated; novel cells are authoritative.

## Recall modes

`act_recall_mode` default **`off`**. Treatment **`separated_key`**. Controls: `raw_p1` (verbatim v35), `early_raw` (raw early `key_rho`, no sparse, no gate), `separated_key_no_familiarity`.

Learning path frozen from v34: awake credit, gated rehearsal, REST, competitive geometry on **P1 values**. n=64. No new η. No auto `cortex.candidate.v36.lock`.
