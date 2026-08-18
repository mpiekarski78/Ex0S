# CORTEX TM044 architecture amendment — learned memory projection

Authorized by [`cortex.candidate.v40.close.lock`](cortex.candidate.v40.close.lock) and [`neural_memory_boundary.lock`](neural_memory_boundary.lock). v40 SOCP stays frozen and off. TM039–TM043 historical locks are not edited. Product **0.0.004**.

This is **not** another ACT recall mode and **not** local synaptic plasticity. It is a learned key/query/value projection into semantically ignorant S.

## Package question

Can the same newborn n=64 cortex learn a cross-context key/query/value projection whose organism-generated query selects an opaque memory record, whose reinstated value causally controls action, and whose persistent store—not runner metadata or cortical weights alone—is necessary for the tested association and revision behavior?

## Projection parameters

\(W_k, W_q, W_v \in \mathbb{R}^{n\times n}\). Optional freeze flag. Not `GenomeConfig`. Birth: \(W_k,W_v\) iid Gaussian from `rng_birth` **after** the existing birth tensors (`v_start`/`v_end` included), so TM044 does not shift earlier geometry. \(W_q\) is initialized equal to \(W_k\) so a query can select a key in the same birth basis. Learning may specialize the three maps. Checkpointed. Load must not require a harness setter.

## Value law

\[
k = W_k\,\rho_{\mathrm{obs}},\qquad
q = W_q\,\rho_{\mathrm{obs}},\qquad
v = W_v\,\rho_{\mathrm{post-credit}}
\]

\(\rho_{\mathrm{obs}}\) is the event-driven cortical state. Credited handle vectors **may** influence \(\rho\) through existing credit. **No handle** is stored as a field or concatenated into \(v\).

Write uses current projections, then (plastic arm only) the organism updates \(W_k,W_q,W_v\) with a local delta rule on internal \(\rho\) and the just-written \(k,v\). No runner loss, gradient, address, or target vector.

## Opaque S

Rows `{key, value, when, provenance_id}`. `when` and `provenance_id` are audit-only and excluded from retrieval. Generic nearest-key on \(q\) only. Ties, empty store, dimensional mismatch, nonfinite → no hit.

## Public path

`event → ρ → q → opaque retrieval → reinstatement of v as scoring address → actuator_decision_scores`

The runner must not compute \(q\), retrieve, or reinstate.

## Arms

`symbolic_oracle` (ceiling), `learned_projection`, `birth_projection` (K/Q/V frozen at birth), `no_persistent_memory`.

Novelty/familiarity is **excluded**. No fitted distance threshold. No `early_raw_half_spacing`. No auto `cortex.candidate.v41.lock`.
