# TM.0.12.MIDPATH results: endpoint provenance vs interior route

**Ex0S under test:** **0.0.003** (not a new stamp)  
**Date:** 15 August 2026  
**Regime:** representation distinguishability (same concrete S; length-4 geometry; projected states only)  
**Run:** `runs/2026-08-15_190539_tm012midpath`  
**Locks:** [`midpath_012.lock`](midpath_012.lock) · pins [`context_012.lock`](context_012.lock) + [`minimap_012.lock`](minimap_012.lock) + [`pathdisc_012.lock`](pathdisc_012.lock) (never rewritten)

`earned_next`: **false** — no Ex0S 0.0.004. No organism / genome change.

## Question

Holding origin X, predecessor P, and frontier Y fixed on one shared S, do two length-4 observed traversals that differ in exactly one interior token collide under H3c `(token, origin, pred)` while distinguishing under H3b (full path)?

C9 proves H3c insufficient only under:

```text
frontier decision information = shared S + projected candidate state
```

Raw apparatus traces are extractor-only ground truth — unavailable after projection.

## Locked cell C9

| Trace | Nodes | Required motor |
|-------|-------|----------------|
| `route_a` | `(X, A, P, Y)` | PRESS (rng pair; must differ) |
| `route_b` | `(X, B, P, Y)` | TUNE |

Geometry: `len==4`; `A[0]==B[0]==X`; `A[1]!=B[1]`; `A[2]==B[2]==P`; `A[3]==B[3]==Y`. All relation `here` and support `(1,0)` identical.

## Computed table

| Candidate | C9 |
|-----------|----|
| H0 token | collision |
| H1 + here | collision |
| H2 + pred | collision |
| H3a + origin | collision |
| H3c + origin+pred | collision |
| H3b + path | D |
| H4 + incoming fid | collision |

D = distinguishes required outputs. Table computed from graph structure (not encoded in the scorer). H3c is **MIDPATH-local** — MINIMAP candidate set unchanged. Outgoing `Y→motor` fids inadmissible; H4 collides on shared `P→Y`.

## Claim

**Endpoint provenance is insufficient.** Some information from the traversed route **interior** is necessary.

- H3c collides on C9 while H3b distinguishes ⇒ no selector whose contextual input is only `(Y,X,P)` can solve both cases from this S.
- **H3b is a surviving candidate, not the product.** Do **not** conclude “store the full path.”
- Ladder: C4 kills predecessor alone; C8 kills origin alone; C9 kills origin+predecessor.
- Next (not this pass): **route-signature minimality** — which interior information can be deleted without required-output collisions.

## Audit notes (apparatus)

Before this recorded run:

1. **`TraceSpec` unpinned** — MIDPATH imported PATHDISC `TraceSpec` without a source SHA (a `trace_len` key could false-positive “pinned”). Fixed: `trace_spec_sha` pin + fail-closed verify.
2. **Duplicate `P→Y` allowed** — H4 collision could be accidental if multiple incoming edges existed. Fixed: validate requires exactly one `P→Y` edge.
3. **Scorer-input test fragile** — string-split could miss kwargs. Tightened to single call-site + explicit projected-state kwargs.
4. PATHDISC lock now records **runtime seed equals locked seed** in validation (lock rewrite → MIDPATH pathdisc pin refreshed).

Scientific table unchanged after fixes.

## Refuse

- Two-store C9; `path_and_frontier` discovery; raw TraceSpec as selector input
- Rewriting CONTEXT / MINIMAP / PATHDISC locks; modifying MINIMAP for H3c
- Declaring full path or a route-signature product this pass
- Genome / cortex / policy; stamping 0.0.004

## Reproduce

```bash
python -m experiments.run_tm012midpath --write-lock   # once
python -m experiments.run_tm012midpath --seed 12345
python tests/test_tm012midpath.py
```
