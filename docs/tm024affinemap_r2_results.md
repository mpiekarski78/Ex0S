# TM.0.24.AFFINEMAP.R2 DEV

Decision: **online_optimization_failure**.

A1 (homogeneous D1) ranks and is robust at four and eight cues. A2 (existing PA) fails four-cue acquire ranking. A3 matches A2 at four and eight cues and is **not** uniquely helpful. Representation is sufficient. Online optimization is the wall. V1 intercept extract stands. Pass statistic is normalized geometric margin. V1 `4a5183e` preserved. SCORE unopened. Product **0.0.004**. `earned_next=false`.

104 unused cells on `TM024.AFFINEMAP.R2.DEV.` / `TWIN.`. Freeze commit `7f2d8e1`. Runner.py SHA `d6289b9b…` unchanged. No neural candidate this package.

## Binding readout

| Question | Result |
| --- | --- |
| Does A1 acquire at 4 and 8? | **Yes** (ranking 4/4) |
| Does A2 acquire at 4 and 8? | **No** (ranking 0/4) |
| Does A3 uniquely help? | **No** (same ranking pattern as A2) |
| Authorized next | **One two-timescale learning candidate** |
| Compact PA path | Not used (A2 failed) |
| Learned actuator bias | Does not join (A3 not uniquely helpful; still unauthorized) |
| Oracle reaudit | Not opened (A1 passed) |
| Another MAP | **No** |

## Capacity (min γ over 4 worlds/orders; twin n=2)

| Arm | Kind | n | rank | pass | pert | min γ | min gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A0 affine D1 | acquire | 2/4/8 | 4/4 | 4/4 | 4/4 | 0.114 / 0.065 / 0.025 | ~2γ |
| A0 | stable | 2/4/8 | 4/4 | 4/4 | 4/4 | 0.115 / 0.064 / 0.025 | ~2γ |
| A1 homogeneous D1 | acquire | 2/4/8 | 4/4 | 4/4 | 4/4 | 0.114 / 0.065 / 0.025 | ~2γ |
| A1 | stable | 2/4/8 | 4/4 | 4/4 | 4/4 | 0.115 / 0.064 / 0.025 | ~2γ |
| A2 PA | acquire | 2 | 4/4 | 4/4 | 0/4 | 0.0100 | 0.020 |
| A2 | acquire | 4 | **0/4** | 0/4 | 0/4 | −0.021 | −0.043 |
| A2 | acquire | 8 | **0/4** | 0/4 | 1/4 | −0.039 | −0.078 |
| A2 | stable | 2 | 4/4 | **0/4** | 0/4 | 0.0097 | 0.019 |
| A2 | stable | 4/8 | 0/4 | 0/4 | 0/4 | −0.020 / −0.012 | |
| A3 bias PA | acquire 4/8 | **0/4** | 0/4 | | same γ as A2 | |

A1/A2 |γ| never exceeded 1. Gap ≈ 2γ on antipodal unit rows. A high pairwise gap with γ<0.01 was not observed on A1; A2 2-cue ranks but misses the 0.01 γ gate after REST and fails perturbation. First PA fail remains **4-cue acquire ranking**.

It is not `affine_intercept_required`, not `apparatus_inconsistency`, not `learned_local_bias_supported`, not `d1_ceiling_reaudit`.

Write-geometry closed. 512/1536 budgets stay closed in this package. Same frozen DEV execution refused.
