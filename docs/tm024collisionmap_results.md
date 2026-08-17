# TM.0.24.COLLISIONMAP results

Primary diagnosis: **`attractor_collapse`**.

Cues A and B separate after cue ingestion, then a single zero-input motor tick makes `ρ_elig` not distinct. Credit therefore lands on effectively the same state for both cues. v29 is frozen. n stays 64. Product **0.0.004**. Amendment not authorized.

## Trace (C0, birth)

| Stage | Cosine | L2 | Distinct? |
| --- | --- | --- | --- |
| start (`v_start`+source) | 1.000 | 0.000 | no |
| cue | 0.712 | 2.613 | **yes** |
| event end | 0.977 | 0.736 | yes |
| observable | 0.999 | 0.180 | yes |
| motor_0 / `ρ_elig` | 0.99986 | 0.048 | **no** |

`sensory_collapse` is false: the cue difference survives `v_end` and the observable-state tick. It dies on the motor loop’s first zero-input tick. That is the state v29 saves as `ρ_elig` and credits.

## Four-way table

| Hypothesis | Fired? | Evidence |
| --- | --- | --- |
| Recurrent/motor-loop attractor collapse | **yes (first match)** | Distinct at cue; not distinct at `ρ_elig`. C1 after teaching A and C2 after A then B show the same motor-tick collapse. C5 twin agrees. |
| Sequential plastic-write interference | no | A/B do not remain distinct at `ρ_elig`, so this arm is skipped. After B, both probes share one ranking. |
| Representation/rank failure | no | C3: 8-cue effective rank at `ρ_elig` is 6; frozen linear separator accuracy 1.0. Tiny residual differences exist; they are not usable as opposing credit states. |
| Plastic update geometry failure | reported, not first | C4 closed-form ridge on two `ρ_elig` columns can interpolate opposite motor vectors (`direct_fit_elig_ok`). v29 sequential teaching cannot (`v29_sequential_ok=false`). With n=64 and two near-duplicates this fit is interpolation, not a usable two-cue geometry. It does not override attractor collapse. |

C2: after A, cue A has beneficial handle score ≈ 1. After B, live A and live B match. Substituting saved teaching `ρ_A` does not recover that one-cue ranking.

## What this is not

Not delayed credit, event-boundary loss, HOLD competition, REST/consolidation, or teacher/probe mismatch (STATEMAP). Not S10/S11 dead age genes. Not an authorization to add two-timescale state, grow n, or run lineage.

A later amendment, if any, must target **motor-loop / zero-input attractor collapse of cue identity before credit**, not a larger cortex and not an L0 function.
