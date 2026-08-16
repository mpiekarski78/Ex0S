# Current organism — Ex0S 0.0.004

A frozen CONTEXT recipe over an inspectable relation graph: provenance-sensitive composition at use time.

Not general intelligence. Not open-ended evolution. Not planning. ACQUIRE/FAMILY show experience can author contextual continuations across generated lives. SKELETON shows experience can also author the relational path those continuations depend on — via **observed-transition acquisition**, not latent map discovery. **Product stamp remains 0.0.004**.

## Claim (defensible — stamped)

> A frozen CONTEXT recipe carries bounded provenance-sensitive state through externally acquired relation graphs and uses that state to distinguish otherwise identical frontiers across unseen generated world families, while acquired continuations remain in S and cognitive weights remain unchanged.

**Lab:** TM.0.13.FAMILY · **Product:** Ex0S 0.0.004 — Contextual Composition  
**Recorded:** [`tm013family_results.md`](tm013family_results.md) · `runs/2026-08-15_223308_tm013family` · **288/288**

Prior stamps still stand: 0.0.003 Frozen Composition ([`genome_011.lock`](genome_011.lock) immutable).

## Recipe files (CONTEXT-on / ACQUIRE-on / SKELETON-on)

| File | Role |
|------|------|
| `three_memory/agent.py` | MATCH, evidence, `_compose_choose` with κ; `use_acquire_ctx`; `observe_symbol` / `use_acquire_skel` |
| `three_memory/kappa.py` | `ksem-sha256-v1` |
| `three_memory/policy.py` | boxed P (`n_feat == 2`) |
| `three_memory/cortex.py` | frozen cortex |
| `experiments/run_tm011compose.py` `make` | compose-on; kwargs forward acquire/skel |

Locks: [`genome_013.lock`](genome_013.lock), [`kappa_013.lock`](kappa_013.lock), [`family_013.lock`](family_013.lock), [`genome_014.lock`](genome_014.lock), [`acquire_014.lock`](acquire_014.lock), [`family_014.lock`](family_014.lock), [`skeleton_015.prereg.lock`](skeleton_015.prereg.lock), [`genome_015.lock`](genome_015.lock), [`skeleton_015.lock`](skeleton_015.lock).

## TM.0.14 → TM.0.15 lineage

| Lab | Result |
|-----|--------|
| ACQUIRE | **16/16** freeze · [`tm014acquire_results.md`](tm014acquire_results.md) |
| FAMILY | **288/288** · `earned_next=true` · **`ex0s=null`** · [`tm014family_results.md`](tm014family_results.md) |
| SKELETON | **16/16** observed-transition · `earned_next=false` · [`tm015skeleton_results.md`](tm015skeleton_results.md) |

0.14: apparatus writes the relation; organism uses it.  
0.15: apparatus emits a symbol sequence; organism writes the relation, then contextual continuation.

## Explicit absences

| Missing | Where next |
|---------|------------|
| Named product stamp for FAMILY earn | human decision (not auto) |
| Latent relation inference from unstructured/ambiguous events | **next interesting attack** |
| Lookahead / backtracking | later |
| No-cue English motor bar | B Fail (untouched) |

## Reproduce

```bash
python tests/test_tm013family.py
python tests/test_tm014acquire.py
python tests/test_tm014family.py
python tests/test_tm015skeleton.py
python -m experiments.run_tm015skeleton --verify-prereg
```

Paper-style summary: [`CLAIM.md`](CLAIM.md).
