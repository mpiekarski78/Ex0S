# TM.0.12.ROUTESIG.DEPTH results: fixed windows vs rolling κ

**Ex0S under test:** **0.0.003** (not a new stamp)  
**Date:** 15 August 2026  
**Regime:** representation-contract (DEPTH family on one shared S)  
**Run:** `runs/2026-08-15_211107_tm012routesig_depth`  
**Locks:** [`routesig_depth_012.lock`](routesig_depth_012.lock) · pins CONTEXT + MINIMAP + PATHDISC + MIDPATH + [`routesig_012.lock`](routesig_012.lock) (ORDER never rewritten)

`earned_next`: **false** — no Ex0S 0.0.004. No organism / genome change. No identity/rename this pass.

## Question

For each fixed window size k∈[1..8], if a route distinction sits **exactly k+1 edges before the frontier**, does suffix-k lose it while suffix-(k+1) and rolling κ retain it?

## Locked family — one S, vary frontier

```text
prefix_a: X → Q → A → Q → B → Q
prefix_b: X → Q → B → Q → A → Q
chain:    Q → T1 → T2 → … → T8
motors:   each T_k → PRESS and T_k → TUNE
```

| Cell | Frontier | Common tail |
|------|----------|-------------|
| C11[k] | `T_k` | `Q→T1→…→T_k` (k edges) |

Motors fixed for all k: route_a → PRESS, route_b → TUNE. Scorer sees projected states + motors only. Geometric assert before scoring: `fa[-k:]==fb[-k:]`, `fa[-(k+1)]!=fb[-(k+1)]`.

## Candidates (DEPTH-local)

| ID | Class |
|----|-------|
| R1 | unordered edge membership at `T_k` |
| R2 | suffix-k |
| R2x | suffix-(k+1) (diagnostic) |
| R3 | ordered route identity |
| R4 | rolling κ `(T_k, κ)` — path-fids only API |

## Computed table (every k = 1..8)

| Candidate | C11[1]…C11[8] |
|-----------|---------------|
| R1 unordered | collision ×8 |
| R2 suffix-k | collision ×8 |
| R2x suffix-(k+1) | distinguishes ×8 |
| R3 ordered | distinguishes ×8 |
| R4 rolling κ | distinguishes ×8 |

C11[2] matches the **C10 suffix-2 outcome pattern** (R1/R2 collide; R3/R4 distinguish). Not the same frozen cell as C10 (`Q→P→Y`).

## Size vs k (careful wording)

| k | R2 fids (in cell k) | R3 fids | R4 bits |
|---|---------------------|---------|---------|
| 1 | 1 | 6 | 256 |
| 2 | 2 | 7 | 256 |
| … | … | … | 256 |
| 8 | 8 | 13 | 256 |

- **R3** grows linearly with k (prefix + common tail).
- **R2** stores exactly k fids **in cell k** — each fixed window forgets the distinction immediately outside it.
- **R4** is one 256-bit κ for every k.

Claim form for R2: **each tested fixed window of size k fails when the relevant distinction occurs k+1 edges back** — not “R2 is constant-sized across the experiment.”

## Claim

**For each tested fixed window k=1..8, a route distinction placed exactly k+1 edges before the frontier is lost by suffix-k but retained by the incremental rolling accumulator.**

**Not:** no finite suffix can ever suffice; SHA-as-genome; full-path necessity; 0.0.004.

## Next (not this pass)

**Identity/rename robustness** of κ. Then freeze accumulator contract; only then discuss CONTEXT genome / 0.0.004.

## Audit notes (apparatus)

After the recorded scientific table:

1. **TraceSpec unpinned** — construction type used without source SHA. Fixed: `trace_spec_sha` + live fail-closed verify.
2. **Outgoing motor refuse one-sided** — only PRESS checked. Fixed: refuse both PRESS and TUNE.
3. **C11[2] pattern soft** — mismatch still returned `ok`. Fixed: fail closed.
4. **Distinct frontiers / path⊆S** — claimed in validation text but not enforced. Fixed in `validate_store`.
5. **Extract / κ live guards** — `path_and_frontier` ban on extract; κ source must not mention TraceSpec/motor; live `score_contrast` SHA.

Scientific table unchanged after fixes.

## Reproduce

```bash
python -m experiments.run_tm012routesig_depth --write-lock   # once
python -m experiments.run_tm012routesig_depth --seed 12345
python tests/test_tm012routesig_depth.py
```
