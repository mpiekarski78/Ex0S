# TM.0.12.ROUTESIG results: route-summary contract (phase 1 — order)

**Ex0S under test:** **0.0.003** (not a new stamp)  
**Date:** 15 August 2026  
**Regime:** representation-contract (summary *classes*, not ad-hoc path tuples)  
**Run:** `runs/2026-08-15_203835_tm012routesig`  
**Locks:** [`routesig_012.lock`](routesig_012.lock) · pins CONTEXT + MINIMAP + PATHDISC + MIDPATH (never rewritten)

`earned_next`: **false** — no Ex0S 0.0.004. No organism / genome change. No ROUTESIG.DEPTH this pass.

## Question

On one shared S, can two observed traces share the **same unordered set of traversed edge fids** while requiring different motors — and does an order-insensitive membership summary collide while ordered / incremental rolling summaries distinguish?

A finite collision table cannot prove “full path storage is necessary” (an opaque digest also distinguishes). This lab asks what properties a valid summary must preserve.

## Locked cell C10 — commutable loops

```text
X → Q;  Q ↔ A;  Q ↔ B;  Q → P → Y → {PRESS, TUNE}
route_a: X → Q → A → Q → B → Q → P → Y → PRESS
route_b: X → Q → B → Q → A → Q → P → Y → TUNE
```

Same unordered path-edge fid set; different ordered sequence. Unique edge per `(bind,did)`. Scorer sees projected states + motors only.

## Candidates

| ID | Class |
|----|-------|
| R0 | endpoint `(Y,X,P)` |
| R1 | unordered edge membership |
| R2 | bounded suffix **k=2** (phase-1 diagnostic) |
| R3 | ordered route identity |
| R4 | rolling κ `(Y, κ)` — path-fids only API |

## Computed table

| Candidate | C10 |
|-----------|-----|
| R0 endpoint | collision |
| R1 unordered edge membership | collision |
| R2 suffix-2 | collision |
| R3 ordered identity | D |
| R4 rolling κ | D |

D = distinguishes. Table computed from graph structure (not encoded in the scorer).

## Claim

**The set of traversed relations is insufficient. Their order is necessary for this locked contrast.** An output-blind incremental order-sensitive accumulator can preserve that distinction without retaining the raw path.

**Not:** SHA-256 is the genome primitive; full path is necessary; rolling hash is minimal; 0.0.004 earned.

## Kappa contract (apparatus)

`kappa_seed(origin)` · `kappa_step(κ, fid)` · `route_kappa(origin, ordered_path_fids)` — never TraceSpec / motor / `context_expect`. Behavioral: same nodes + different motors ⇒ same κ. Shuffle-invariant under unique edges.

## Next (not this pass)

**ROUTESIG.DEPTH** — systematic bounded-suffix family. Then freeze accumulator requirements; only then discuss compose carrying `(Y, κ)`.

## Audit notes (apparatus)

Before this recorded run:

1. **Global directed-edge uniqueness** — only traversed hops called `edge_fid`; a duplicate `(bind,did)` elsewhere in S could still poison file-order stories. Fixed: every directed pair in S must appear exactly once.
2. **Path-edge multiset** — frozenset equality alone allows multiplicity mismatches. Validate now also requires equal path-edge fid **multisets**.
3. **`TraceSpec` unpinned** — imported for cell construction without a source SHA. Fixed: `trace_spec_sha` + fail-closed verify.
4. **κ output-blind test** — tightened to assert `extract_states_routesig` R4 identical when only `TraceSpec.required_motor` differs.
5. Live SHA fail-closed for `score_contrast` / `edge_fid` / `route_kappa` / `TraceSpec`.

Scientific table unchanged after fixes.

## Reproduce

```bash
python -m experiments.run_tm012routesig --write-lock   # once
python -m experiments.run_tm012routesig --seed 12345
python tests/test_tm012routesig.py
```
