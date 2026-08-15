# Current organism — Ex0S 0.0.003

A frozen, evidence-aware interpreter over an inspectable relation graph.

Not general intelligence. Not open-ended evolution. Not planning.

**Product stamp stays 0.0.003.** TM.0.13.CONTEXT installs a CONTEXT-on **candidate** (`use_context_kappa`) under test — not stamped as Ex0S 0.0.004.

## Claim (defensible — stamped)

A fixed procedural machine can store world-specific relations outside its weights, causally depend on that store after ρ reset, revise those relations, and compose them without materializing shortcuts.

## CONTEXT candidate (not stamped)

Under `use_context_kappa=True`, M carries transient κ after selected non-motor hops and filters derived frontiers by `ctx`. Locked small family: [`tm013context_results.md`](tm013context_results.md). Locks: [`kappa_013.lock`](kappa_013.lock), [`genome_013.lock`](genome_013.lock), [`context_013.lock`](context_013.lock). `earned_next` false.

## Recipe files (0.0.003 historical freeze)

| File | Role |
|------|------|
| `three_memory/agent.py` | MATCH, evidence, `_compose_choose`, outcome handling |
| `three_memory/policy.py` | boxed P (`n_feat == 2`) |
| `three_memory/cortex.py` | frozen cortex |
| `experiments/run_tm011compose.py` `make` | compose-on constructor (`use_context_kappa=False`) |

Lock: [`genome_011.lock`](genome_011.lock) — **immutable**; do not rewrite for CONTEXT.  
Baseline commit: `c392aa515b7a3445bb15bc55ad969d971632ea3f`  

Verify:

- **Historical:** recipe blobs at baseline commit match `genome_011.lock`
- **Compatibility:** HEAD with `use_context_kappa=False` still hosts 0.0.003 compose behavior

Rewrite the 0.0.003 lock only with:

```bash
python -m experiments.run_tm011family --write-lock --baseline-commit <sha>
```

Tests must not write this lock. CONTEXT-on hashes live in `genome_013.lock`.

## What 252/252 means

The FAMILY battery planted structured `.tag` relations and reported outcomes onto selected fact IDs. The frozen mechanism **operated over 252 generated external-state worlds**. It did not autonomously acquire those relations from raw events. With CONTEXT-on candidate on HEAD, report **behavioral compatibility 252/252** (feature off); do not claim `genome_delta=0` for HEAD vs `genome_011.lock`.

## Explicit absences

| Missing | Where documented |
|---------|------------------|
| Stamped provenance-sensitive Ex0S | TM.0.13.CONTEXT candidate only; **TM.0.13.FAMILY** next |
| Lookahead / backtracking | TM.0.11.BOUND `local_optimum_dead_end` |
| No-cue English motor bar | B Fail (untouched) |
| Autonomous acquisition from open experience | not yet |

## Reproduce freeze + unit checks

```bash
python tests/test_tm011family.py
python tests/test_tm011bound.py
python tests/test_tm012context.py
python tests/test_tm012minimap.py
python tests/test_tm012pathdisc.py
python tests/test_tm012midpath.py
python tests/test_tm012routesig.py
python tests/test_tm012routesig_depth.py
python tests/test_tm012routesig_identity.py
python tests/test_tm013context.py
```

Paper-style summary: [`CLAIM.md`](CLAIM.md).
