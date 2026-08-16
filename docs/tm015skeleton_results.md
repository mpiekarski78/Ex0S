# TM.0.15.SKELETON results — observed-transition acquisition

**Ex0S under test:** 0.0.004 (product stamp unchanged)  
**Lab:** TM.0.15.SKELETON  
**Date:** 2026-08-16  
**Recorded:** `python -m experiments.run_tm015skeleton --write-lock` → **16/16**  
**Earn:** `earned_next=false`, **`ex0s=null`**

**Locks:** [`skeleton_015.prereg.lock`](skeleton_015.prereg.lock) (wording freeze) · [`genome_015.lock`](genome_015.lock) · [`skeleton_015.lock`](skeleton_015.lock)

## Audit hardenings (this pass)

- D7: equal-support HOLD requires `evidence_tie` (removed vacuous `or True`); `traverse_hold` surfaces the flag
- D11: real lexicographic filename permutation (TagStore loads sorted)
- D12: teacher contract pins `set(info.keys()) != {"action"}` like ACQUIRE
- D1: graph edges must equal organism `experience_skel` only (no co-planted apparatus skeleton)
- D6: swapped `lived_kappa` must match B-/A-route oracles (not HOLD-only)
- D5: strip-skel asserts `compose_hops == 0`
- Ledger: unexpected provenance audited across all cell stores before earn
- `_hold()` always clears/sets `compose_hops` (including 0)

## Label (binding)

**Observed-transition acquisition** — not latent relation discovery.

Apparatus emits a temporal symbol sequence `[X, A, Y]`. The organism keeps one transient `prev` and authors adjacency into S. The observed sequence supplies transition structure; Ex0S has **not** inferred latent relations from unstructured sensory experience.

## Preregistered claim

> A frozen developmental recipe can convert an observed sequence of relational transitions into durable skeleton edges in S, then compose over those organism-authored edges to rebuild κ and author provenance-sensitive contextual continuations, without the apparatus writing the skeleton or contextual answers into S.

## Headline

| | |
|--|--|
| Battery | **16/16** |
| Birth | X cannot reach Y; no `experience_skel` |
| After symbols | organism-authored `X→A`, `A→Y` (`source=experience_skel`) |
| After teacher | `Y→PRESS ctx=κA` (`source=experience_ctx`) |
| Dual-strip | skel-only → HOLD; ctx-only → HOLD; restore → PRESS |
| Unexpected skel/ctx writes | **0** |
| Product stamp | **none** |

## What 16/16 means

Experience now creates both the durable relational path and the contextual continuation that later depends on that path.

Still explicitly: the observed sequence supplies transition structure; Ex0S has not inferred latent relations from unstructured sensory experience.

## Progression

| Stage | Apparatus | Organism |
|-------|-----------|----------|
| 0.14 | writes relation into S | uses relation; authors `experience_ctx` |
| **0.15** | emits symbol sequence | writes relation (`experience_skel`); then `experience_ctx` |
| later | less-structured / ambiguous events | infer entities, transitions, what to store |

## Shows / does not show

**Shows:** `observe_symbol` adjacency rule; reachability from organism-authored skeleton; compose κ over authored edges; ACQUIRE contextual continuation without planted map; dual-strip causality; competing support without invented contradiction; stale `prev` cleared by `reset_ρ`.

**Does not show:** latent map discovery; open-ended exploration; LOOKAHEAD; a named Ex0S product bump; FAMILY-scale generalization of skeleton acquisition.

## Next

Unstructured / ambiguous sensory events (which entities? which transition? which relation deserves storage?). Not LOOKAHEAD. Not another FAMILY this pass. Product stamp naming remains a human decision.

## Reproduce

```bash
python -m experiments.run_tm015skeleton --verify-prereg
python tests/test_tm015skeleton.py
python -m experiments.run_tm015skeleton --write-lock
```
