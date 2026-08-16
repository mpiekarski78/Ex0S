# TM.0.16.ALIASWALL — cross-episode alias equivalence wall

**Lab:** TM.0.16.ALIASWALL (bookkeeping on frozen RELATE; not a product stamp)  
**Product under test:** Ex0S **0.0.004**  
**Recorded:** `python -m experiments.run_tm016aliaswall --write-lock` → **6/6**  
**Flags:** `earned_next=false`, `ex0s=null`

## Claim

> With RELATE fixed, replacing repeated route symbols with fresh opaque aliases disperses support across disjoint edges; no reusable relation survives, so compose remains HOLD despite repetition of the same latent route.

## Bounded fact

Frozen RELATE succeeds only when symbol equivalence is supplied externally; repetition of latent structure alone does not currently produce reusable relations.

This does **not** claim aliases are learnably equivalent. Object continuity / gap persistence is untouched. **Not TM.0.17.**

## Paired worlds

| World | Route tokens | Result |
|-------|----------------|--------|
| Control | stable `x,a,y` | unique winners; `lived_bind=y` |
| Kill | pinned opaque aliases (`kelm/norb/wift`, …) | support=1 per mapped instance; motor HOLD |

Each cell: fresh `make_relate`, fresh `UsePolicy`, empty S, reset ρ. All six S directories are distinct. W1 also asserts support=0 for every edge that would appear if an episode boundary leaked across lives.

## Battery

| Cell | Result |
|------|--------|
| W0 Control | OK |
| W1 Kill (exact invariants) | OK |
| W2 schedule twin (canonicalize) | OK |
| W3 opacity | OK |
| W4 map isolation | OK |
| W5 no mechanism | OK |

## Locks

- Prereg: [`alias_wall.prereg.lock`](alias_wall.prereg.lock) (pins `genome_016.lock` + `relate_016.lock`; no ALIASWALL artifact SHAs)
- Freeze: [`alias_wall.lock`](alias_wall.lock)

## Reproduce

```bash
python -m experiments.run_tm016aliaswall --verify-prereg
python tests/test_tm016aliaswall.py
python -m experiments.run_tm016aliaswall --write-lock
```

## Next (not this pass)

Decide what observable evidence could legitimately establish alias equivalence; then earn it. Separately: gap persistence / object continuity. Later: anonymous features / encoders.
