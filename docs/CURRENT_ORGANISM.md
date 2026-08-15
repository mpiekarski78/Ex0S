# Current organism — Ex0S 0.0.003

A frozen, evidence-aware interpreter over an inspectable relation graph.

Not general intelligence. Not open-ended evolution. Not planning.

## Claim (defensible)

A fixed procedural machine can store world-specific relations outside its weights, causally depend on that store after ρ reset, revise those relations, and compose them without materializing shortcuts.

## Recipe files (must not drift)

| File | Role |
|------|------|
| `three_memory/agent.py` | MATCH, evidence, `_compose_choose`, outcome handling |
| `three_memory/policy.py` | boxed P (`n_feat == 2`) |
| `three_memory/cortex.py` | frozen cortex |
| `experiments/run_tm011compose.py` `make` | compose-on constructor |

Lock: [`genome_011.lock`](genome_011.lock)  
Baseline commit: `c392aa515b7a3445bb15bc55ad969d971632ea3f`  
Verify: `python -m experiments.run_tm011family` fails closed if any recipe SHA, including `agent_sha`, moves.

Rewrite the lock only with:

```bash
python -m experiments.run_tm011family --write-lock --baseline-commit <sha>
```

Tests must not write this lock.

## What 252/252 means

The FAMILY battery planted structured `.tag` relations and reported outcomes onto selected fact IDs. The frozen mechanism **operated over 252 generated external-state worlds**. It did not autonomously acquire those relations from raw events.

## Explicit absences

| Missing | Where documented |
|---------|------------------|
| Contextual / provenance-sensitive composition | TM.0.12.CONTEXT + MINIMAP — origin least-structured candidate on C2–C6; C7 unobservable |
| Lookahead / backtracking | TM.0.11.BOUND `local_optimum_dead_end` |
| No-cue English motor bar | B Fail (untouched) |
| Autonomous acquisition from open experience | not yet |

## Reproduce freeze + unit checks

```bash
python tests/test_tm011family.py
python tests/test_tm011bound.py
python tests/test_tm012context.py
```

Paper-style summary: [`CLAIM.md`](CLAIM.md).
