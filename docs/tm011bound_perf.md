# TM.0.11.BOUND performance note (infrastructure only)

**Not** part of TM.0.12.CONTEXT. Do not mix latency work with provenance genomes.

## Observation (BOUND recorded run)

From `runs/2026-08-15_163528_tm011bound` (within-model throughout):

| Depth | \|S\| | Wall ms (order) |
|-------|------|-----------------|
| 2 | 10 | ~0.3 |
| 2 | 100 | ~4–5 |
| 2 | 1000 | ~600+ |
| 5 | 10 | ~0.3 |
| 5 | 100 | ~4–5 |
| 5 | 1000 | ~300+ |

A 100× increase in \|S\| producing roughly 2000× latency at depth 2 is uglier than linear scan. Profile before considering GPU.

## Profile separately

Measure wall time / counts for:

1. Store enumeration (list / glob)
2. File reads / tag parsing
3. MATCH
4. EVIDENCE
5. Sorting / ranking
6. Hashing / instrumentation
7. Per-hop repeated full-store scans

## Optimization rule

Index or cache only under **behavioral-equivalence regression**:

```text
candidate set (old retrieval)  ==  candidate set (indexed retrieval)
```

on frozen compose worlds (FAMILY / BOUND within-model cells). Motors and hops must match. Cognition claim unchanged.

LOOKAHEAD and CONTEXT remain separate scientific tracks; this note is infrastructure only.
