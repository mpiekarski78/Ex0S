# TM.0.11.BOUND results: envelope of frozen Ex0S 0.0.003

**Ex0S under test:** **0.0.003** (not a new stamp)  
**Date:** 15 August 2026  
**Regime:** capacity / conceptual bound (not a recipe jump)  
**Run:** `runs/2026-08-15_163528_tm011bound`  
**Locks:** [`genome_011.lock`](genome_011.lock) (organism) · [`bound_011.lock`](bound_011.lock) (apparatus, preregistered)

`earned_next`: **false** — no Ex0S 0.0.004.

## Headline

| Bucket | Result |
|--------|--------|
| Within-model cells | **42/42** pass |
| Expected boundaries | **2/2** confirmed |
| Unexpected Fail | **0** |
| Resource Boundary | **0** (timeout 30s) |
| Genome / apparatus interventions | **0 / 0** |

## Four-class summary

| Class | Count |
|-------|-------|
| Within-model Pass | 42 |
| Expected Boundary | 2 |
| Unexpected Fail | 0 |
| Resource Boundary | 0 |

## Curves (honest)

**Depth 2–20:** all Within-model Pass (hold-out 16 and 20 included). No hop flag.

**\|S\| strata** (latency rises; correctness holds):

| Depth | \|S\| | Class | Wall ms (order) | facts_examined |
|-------|------|-------|-----------------|----------------|
| 2 | 10 | within | ~0.3 | 20 |
| 2 | 100 | within | ~4–5 | 200 |
| 2 | 1000 | within | ~600+ | 2000 |
| 5 | 10 | within | ~0.3 | 50 |
| 5 | 100 | within | ~4–5 | 500 |
| 5 | 1000 | within | ~300+ | 5000 |

**Branch 1–8 + first-hop tie:** Within-model.  
**Off-path distractors 0–500:** Within-model (wrong-motor junk).  
**Revise-far 3 / 5 / 8:** Within-model; upstream hashes stable.  
**Nasty (dirty S + revise D):** Within-model Pass.  
**Cycles / no-motor / deep tie / file_order:** Within-model.

## Expected boundaries (preregistered)

| Cell | Expected | Confirmed |
|------|----------|-----------|
| `local_optimum_dead_end` | HOLD — no lookahead/backtracking | yes |
| `reuse_a` | strong **wrong-`here`** path wins — compose is bind-only | yes |

These are **not** failures of 0.0.003. They document machinery the frozen organism does not contain.

## Audit notes (apparatus)

Before this recorded run:

1. **`reuse_a` was non-discriminating** — strong evidence sat on the *probe* `here`, so here-filter and bind-only both predicted the same motor. Fixed: strong evidence on wrong-`here` (`cha`), weak on probe `here` (`chb`). Hold-out generators unchanged.
2. **`file_order` cannot stress unsorted iteration** — `TagStore` loads sorted by filename. Kept as content-invariance under Rel-list shuffle; annotation corrected.
3. **Probe timeout** — each probe now runs under `ThreadPoolExecutor` with preregistered `timeout_ms` (not only a post-phase wall check).

## Reading

0.0.003’s **semantic** envelope covered this sweep: depth through 20, \|S\| through 1000, branching, dirty revise. The **computational** envelope shows clear \|S\| latency growth (performance degradation, still Within-model). The **conceptual** envelope is the two Expected Boundaries: no backtracking from a locally preferred dead end, and no `here`-filter on compose frontiers.

## What this earns / What next

BOUND strengthens the 0.0.003 claim inside its semantics and makes the absences crisp rather than inventing the next recipe from a shopping list.

| Track | Status |
|-------|--------|
| **TM.0.12.CONTEXT** | Next — representation audit (which provenance? how much?). Not a genome patch; not 0.0.004. |
| **LOOKAHEAD** | Later, separate battery — do not mix with CONTEXT (lookahead can brute-force continuations). |
| **\|S\| latency** | Infra only — profile / index under behavioral-equivalence regression; keep out of CONTEXT. |

Do not pre-name a future stamp “Abstraction.” If a minimal structured CONTEXT primitive later freezes and generalizes, the empirically justified candidate claim is **context-preserving composition**.

## Reproduce

```bash
python -m experiments.run_tm011bound --write-lock   # once, before hold-outs
python -m experiments.run_tm011bound --seed 12345 --workers 4
```
