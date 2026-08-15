# TM.0.12.PATHDISC results: same-S origin vs path

**Ex0S under test:** **0.0.003** (not a new stamp)  
**Date:** 15 August 2026  
**Regime:** representation distinguishability (same concrete S; apparatus-observed traces)  
**Run:** `runs/2026-08-15_184916_tm012pathdisc`  
**Locks:** [`pathdisc_012.lock`](pathdisc_012.lock) · pins [`context_012.lock`](context_012.lock) + [`minimap_012.lock`](minimap_012.lock) (never rewritten)

`earned_next`: **false** — no Ex0S 0.0.004. No organism / genome change.

## Question

MINIMAP’s collision argument is valid only when two probes share the **same concrete S**. Holding origin and frontier fixed, do two **observed** traversals that require different motors collide under H3a while distinguishing under H3b?

Two stores with the same `(Y,X)` do **not** kill H3a (false lower bound). Inferring a diamond route with `path_and_frontier` recreates C7 (experimenter-private answer).

## Locked cell C8

One shared diamond S; two apparatus traces:

| Trace | Nodes | Required motor |
|-------|-------|----------------|
| `route_a` | `(X, A, B, Y)` | PRESS (or rng pair; must differ) |
| `route_b` | `(X, C, D, Y)` | TUNE |

Candidates are projections via `extract_states_from_trace` — **not** route discovery.

## Computed table

| Candidate | C8 |
|-----------|----|
| H0 token | collision |
| H1 + here | collision |
| H2 + pred | D |
| H3a + origin | collision |
| H3b + path | D |
| H4 + incoming fid | D |

D = distinguishes required outputs. Table computed from graph structure (not encoded in the scorer). Outgoing `Y→motor` fids are answer-derived and **inadmissible**.

## Claim

**Origin alone is insufficient.** Some information about the traversed route **beyond origin** is necessary.

- H3a collides on C8 while H3b distinguishes ⇒ no selector whose contextual input is only H3a can solve both cases from this S.
- **H3b is a surviving candidate, not the product.** H2 also distinguishes on C8 — do **not** conclude “store the full path.”
- C4 already kills predecessor alone; C8 kills origin alone.
- Next question (not this pass): **H3c = (token, origin, predecessor)** vs richer path signatures.

## Audit notes (apparatus)

Before this recorded run:

1. **Y→motor validation gap** — `validate_c8_pair` accepted traces whose required motors lacked outgoing edges (or were not in `MOTORS`). Fixed: fail closed on missing `Y→motor` / unknown motor.
2. **Vacuous same-S check** — hashed the same list against itself. Fixed: both traces' path + motor edge fids must be non-empty subsets of one shared store.
3. **Scorer drift hole** — PATHDISC imported live `score_contrast` while only pinning `minimap_012.lock` bytes; scorer edits without a minimap lock rewrite would silently apply. Fixed: pin + verify `scorer_sha`.
4. **Weak 0.0.004 test** — allowed `"0.0.004" in blob or earned_next is False` (always true once refuse mentions the stamp). Tightened to `earned_next is False` + refuse entry.

Scientific table unchanged after fixes.

## Refuse

- Two-store C8; discovering routes with `path_and_frontier`
- Rewriting CONTEXT / MINIMAP locks or C0–C7
- Declaring full path or H3c the product this pass
- Genome / cortex / policy; probing 0.0.003; stamping 0.0.004

## Reproduce

```bash
python -m experiments.run_tm012pathdisc --write-lock   # once
python -m experiments.run_tm012pathdisc --seed 12345
python tests/test_tm012pathdisc.py
```
