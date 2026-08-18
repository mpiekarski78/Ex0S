# TM.0.62.XGEN results

## Decision: `only_unrestricted_interpolates`

Product **0.0.004**. Canonical law unchanged. Floor 0.05 was not retuned.
Transport fitted on 8 chronological action-complete pairs and scored on 8 later held-out contexts.
Every W* and every transport map was discarded. TM061 remains valid and narrow.

Unrestricted linear interpolates its 8 fitting pairs at 8/8 with ~0 residual. Identity, orthogonal, diagonal, affine, rank-4, ridge (λ=1), and minimum-norm maps all fail the held-out later contexts. Per-action maps do not generalize across all four roles. Design matrix rank is 8 in 64-D (condition ~90–97). TM061's in-sample 16/16 was interpolation, not a learnable generic transport law.

Return to representation-learning hypotheses. This is not an architecture, not an installed map, and not a Miconi-style jump.

## Ceilings (n_ok / n_need, both worlds)

| Map | Fit pairs | Held-out later contexts |
| --- | --- | --- |
| Unrestricted in-sample linear | 8/8, 8/8 | not scored as a generalization claim |
| Identity | — | 3/8 and 4/8 forward |
| Orthogonal | — | 1/8 and 2/8 forward |
| Diagonal | — | 2/8, 2/8 |
| Affine | — | 2/8, 2/8 |
| Low-rank (4) | — | 1/8, 2/8 |
| Ridge λ=1 (dof ≈ 0.97) | — | 2/8, 2/8 |
| Minimum-norm linear | — | 2/8, 2/8 |
| Action-conditioned min-norm | — | not all four roles |

Held-out prefix→later L2 is ≈0.49 for every shared holdout map versus ≈0 for in-sample interpolation.

## What this is not

Not an installed oracle, not a neural edit, not a K/Q/V redesign, not a v41 candidate,
not a generic transport law claimed from TM061 in-sample 16/16.
