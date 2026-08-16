# Current organism — Ex0S 0.0.004

A frozen CONTEXT recipe over an inspectable relation graph: provenance-sensitive composition at use time.

Not general intelligence. Not open-ended evolution. Not planning. ACQUIRE/FAMILY show experience can author contextual continuations across generated lives. SKELETON shows experience can author the relational path from a symbol stream. RELATE shows ambiguous multi-symbol events can accumulate a candidate cloud and let converging evidence select which acquired relation **controls behavior** (losers stay in S; focus unused). **Product stamp remains 0.0.004**.

## Claim (defensible — stamped)

> A frozen CONTEXT recipe carries bounded provenance-sensitive state through externally acquired relation graphs and uses that state to distinguish otherwise identical frontiers across unseen generated world families, while acquired continuations remain in S and cognitive weights remain unchanged.

**Lab:** TM.0.13.FAMILY · **Product:** Ex0S 0.0.004 — Contextual Composition  
**Recorded:** [`tm013family_results.md`](tm013family_results.md) · `runs/2026-08-15_223308_tm013family` · **288/288**

Prior stamps still stand: 0.0.003 Frozen Composition ([`genome_011.lock`](genome_011.lock) immutable).

## Recipe files (CONTEXT-on / ACQUIRE-on / SKELETON-on / RELATE-on)

| File | Role |
|------|------|
| `three_memory/agent.py` | MATCH, evidence, `_compose_choose` with κ; `use_acquire_ctx`; `observe_symbol` / `use_acquire_skel`; `observe_event` / `end_event_episode` / `use_acquire_relate` |
| `three_memory/kappa.py` | `ksem-sha256-v1` |
| `three_memory/policy.py` | boxed P (`n_feat == 2`) |
| `three_memory/cortex.py` | frozen cortex |
| `experiments/run_tm011compose.py` `make` | compose-on; kwargs forward acquire/skel/relate |

Locks: [`genome_013.lock`](genome_013.lock), [`kappa_013.lock`](kappa_013.lock), [`family_013.lock`](family_013.lock), [`genome_014.lock`](genome_014.lock), [`acquire_014.lock`](acquire_014.lock), [`family_014.lock`](family_014.lock), [`skeleton_015.prereg.lock`](skeleton_015.prereg.lock), [`genome_015.lock`](genome_015.lock), [`skeleton_015.lock`](skeleton_015.lock), [`relate_016.prereg.lock`](relate_016.prereg.lock), [`genome_016.lock`](genome_016.lock), [`relate_016.lock`](relate_016.lock), [`alias_wall.prereg.lock`](alias_wall.prereg.lock), [`alias_wall.lock`](alias_wall.lock).

## TM.0.14 → TM.0.16 lineage

| Lab | Result |
|-----|--------|
| ACQUIRE | **16/16** freeze · [`tm014acquire_results.md`](tm014acquire_results.md) |
| FAMILY | **288/288** · `earned_next=true` · **`ex0s=null`** · [`tm014family_results.md`](tm014family_results.md) |
| SKELETON | **16/16** observed-transition · `earned_next=false` · [`tm015skeleton_results.md`](tm015skeleton_results.md) |
| RELATE | **16/16** candidate relations under ambiguity · `earned_next=false` · [`tm016relate_results.md`](tm016relate_results.md) |
| ALIASWALL | **6/6** Control vs opaque-alias Kill on frozen RELATE · `earned_next=false` · [`tm016aliaswall_results.md`](tm016aliaswall_results.md) |

0.14: apparatus writes the relation; organism uses it.  
0.15: apparatus emits a symbol sequence; organism writes the relation, then contextual continuation.  
0.16: apparatus emits ambiguous multi-symbol events; organism writes a candidate cloud; evidence selects which relation controls compose (losers stay in S).  
ALIASWALL: same latent route with fresh opaque aliases → support fragments; compose HOLD — symbol equivalence is still external.

## Explicit absences

| Missing | Where next |
|---------|------------|
| Named product stamp for FAMILY earn | human decision (not auto) |
| Cross-episode alias equivalence (earn) | after observable evidence is specified — wall frozen in [`tm016aliaswall_results.md`](tm016aliaswall_results.md) |
| Gap persistence / object continuity | later, separate from alias equivalence |
| Anonymous features / sensory encoders / pixels | later |
| Lookahead / backtracking | later |
| No-cue English motor bar | B Fail (untouched) |

## Reproduce

```bash
python tests/test_tm013family.py
python tests/test_tm014acquire.py
python tests/test_tm014family.py
python tests/test_tm015skeleton.py
python tests/test_tm016relate.py
python tests/test_tm016aliaswall.py
python -m experiments.run_tm016aliaswall --verify-prereg
```

Paper-style summary: [`CLAIM.md`](CLAIM.md).
