# CORTEX v37 architecture amendment

Authorized by [`lineage_keygeom.closure.lock`](lineage_keygeom.closure.lock). TM030 / v35 / TM029 historical locks are **not** edited.

TM030 showed early `key_rho` retrieval identity is stable under history and perturbation. The fixed Hadamard + k-WTA separator added collisions and false familiarity. v36 killed event-end P1 retrieval specifically (`raw_p1`); it did not kill raw retrieval generally (`early_raw` passed taught-stable probes).

v37 tests the **hypothesis** that early-raw episodic indexing plus a **geometry-derived conservative familiarity rule** can reject novelty without learned separation and without a DEV-derived numeric cutoff.

## Dual path (required)

1. **Write/replacement unchanged:** eight content-addressed P1 episodes; L2 radius **0.05**; contradictory replacement. Do not match or replace on `key_rho` distance.
2. **Recall is the only new path.** Never return stored handle. Canonical motor API remains `actuator_decision_scores`.
3. **No Hadamard / k-WTA on the treatment path.** Sparse `separated_key` is a matched retrieval-path control on treatment-trained state, not a v36 train reproduction. Historical TM029 remains the failed-v36 control.

## Familiarity (parameter-free)

From current valid keyed slots:

\[
R = \tfrac{1}{2} \min_{i \neq j} \|k_i - k_j\|_2
\]

Accept unique nearest stored `key_rho` iff \(d_1 \le R\) (strict IEEE compare; no epsilon). Exact nearest-distance ties → cortical fallback. Fewer than two keyed episodes, duplicate stored keys, or \(R=0\) → cortical fallback. Missing live or stored `key_rho` → cortical fallback.

This guarantees **non-overlapping accepted balls**. It does **not** mathematically guarantee novelty rejection: a novel key can still fall inside one ball. DEV decides whether the rule works as a novelty signal on new geometries.

Record `R`, `d1`, `d2`, pairwise minimum \(B\), the two slots realizing that minimum, and fallback `reason`. One near-duplicate pair can shrink \(R\) for the whole store.

## Recall modes

`act_recall_mode` default **`off`**. Treatment **`early_raw_half_spacing`**. Matched clones on a treatment-trained checkpoint:

| mode | isolates |
|------|----------|
| `early_raw_half_spacing` | treatment |
| `early_raw` | removal of the gate |
| `off` | whether recall is necessary |
| `separated_key` | old sparse transform on identical learned state |

## Replicates

Four `seed_registry` values only. All other `GenomeConfig` fields identical. Token renaming is not a geometry replicate. Independent first-OOV samples use pinned dummy `_vocab_vec` skips, not four spellings under one registry state.

Learning path frozen from v34/v36 write law. n=64. No new η. No auto `cortex.candidate.v37.lock`. Product **0.0.004**.
