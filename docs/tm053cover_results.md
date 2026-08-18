# TM.0.53.COVER results

## Decision: `coverage_generalizes`

Product **0.0.004**. No v41 lock. W* not installed. Existing joint SOCP only. **N is not an organism constant.**
TM052 first-match `shared_W_star_satisfies` and addendum `sampled_capacity_without_generalizable_grounding` are unchanged. Held-out cues were never SOCP constraints.

DEV ran once on clean `721fc8a7cbd5be8b96012bfbe602b4cbaa63d511`. Frozen runner SHA `62e2fe15a3e0565d9041c36cfb16b7fae24d98c8211dec9358a0437770dd2bb4`. 42 cells (6 setup + 36 scored). Three registry seeds, two fresh world families each.

The frozen ladder fires `coverage_generalizes` because at N=32 every seed and world is 16/16 on held-out wraps, 4/4 on references, and feasible. That predicate is satisfied. It is **not** a coverage-growth effect.

## Load-bearing geometry

Wrapped P1s collapsed by action. Independent cue strings did not make independent states.

- Unique train hashes at N=32: **4** (one per action), not 128
- Unique hold hashes: **4**, and they are the same four
- Within-action L2: **0**
- Between-action L2 mean ≈ **0.042**, still under the 0.05 episode-match floor
- Frobenius movement from parent W is ≈ **0.095** at every N (duplicate constraints do not move the solution)

N=1 is already 4/4 train, 16/16 hold, 4/4 refs. The table is flat because there is no new experience to add.

## Curve (mean across seeds × worlds)

| N | Train | Hold | Hold4 | Refs | Feasible |
| --- | --- | --- | --- | --- | --- |
| 1 | 1.00 | 1.00 | 1.00 | 1.00 | 6/6 |
| 2 | 1.00 | 1.00 | 1.00 | 1.00 | 6/6 |
| 4 | 1.00 | 1.00 | 1.00 | 1.00 | 6/6 |
| 8 | 1.00 | 1.00 | 1.00 | 1.00 | 6/6 |
| 16 | 1.00 | 1.00 | 1.00 | 1.00 | 6/6 |
| 32 | 1.00 | 1.00 | 1.00 | 1.00 | 6/6 |

## Reading

This does **not** show that broader generic experience generalizes to TM052’s distinct held-out wraps. Those TM052 hold states had different hashes from the training writes. Here the held-out wraps are copies of the training wraps, so transfer is tautological.

Generic consolidation remains plausible only in the weak sense already given by TM052: a shared W* exists on a small set of action-locked wrap states plus references. It is not demonstrated as a learning curve.

Do not install W*. Do not take N=1 or N=32 as an organism constant. Do not extend v40. The feedback wrap still lacks context diversity on this family.

## What this is not

Not an installed oracle, not a fitted N, not a second decoder, not a v41 candidate, not product 0.0.005. TM052 remains diagnostic only.
