# TM.0.44.MEMPROJ contract

**Lab:** TM.0.44.MEMPROJ · **Not a v41 candidate.** Product **0.0.004**.

v40 SOCP stays frozen and off. Novelty/familiarity is excluded. Renamed worlds are independent renamed-symbol replicates, not zero-shot aliases.

## Arms

- `symbolic_oracle` — current episode/ACT path. Ceiling only.
- `learned_projection` — plastic \(W_k,W_q,W_v\); opaque S.
- `birth_projection` — opaque S; K/Q/V frozen at birth.
- `no_persistent_memory` — plastic cortex; no persistent rows.

## Value

\(v=W_v\rho_{\text{post-credit}}\). No handle in the record.

## Donor

Projection-matched, mapping-naive host. Transplant opaque rows only. Hash mismatch is `donor_basis_mismatch`.

## API

Runner calls `event_memory_scores` after events. It does not compute \(q\), retrieve, or reinstate.
