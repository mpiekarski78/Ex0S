# TM.0.24.PHASEMAP DEV

Decision: **state_reinstatement_instability**. First fail: **P2**. Last robust eight-cue phase: **P1** (event-end).

Eight-cue robust: `P0=true`, `P1=true`, `P2=false`, `P3=false`, `P4=false`, `P5=false`.

Write-geometry closed. SCORE unopened. No neural candidate. 1536 eligibility budget stays closed. Product **0.0.004**. `earned_next=false`.

## What was measured

84 cells on unused `TM024.PHASEMAP.DEV.` / `TWIN.` (72 rank + 12 renamed twins). Same organism captured at P0–P5. Frozen DISCRIMMAP R2 D1 is the only oracle. Fit teaching select-tick phase rows only. Probes, extra repeats, and perturbations are never fit.

Pass requires D1 `optimal`, train and probe sign, train and probe \(\gamma\ge 0.01\), and perturbation (`σ=0.01`, `n=20`, `≥19/20`). Robust at a phase means that rule on all eight-cue rank cells and the two-cue twins.

Contraction \(C_t=\|\rho_{t+1}^A-\rho_{t+1}^B\|/\|\rho_t^A-\rho_t^B\|\). \(C_t<1\) is contraction.

## Result

P0 (cue) and P1 (event-end) are robust at two, four, and eight cues, including renamed twins. Teaching at P2 is still hard-margin clean (`optimal`, train \(\gamma\approx 0.0102\)), but same-phase live probes fail (probe \(\gamma\) at or below zero; perturbation 0/4). That is the first-match ladder code.

P2 and P3 are identical (`C_{P2\to P3}=1`). P4 and P5 are identical (`C_{P4\to P5}=1`). There are two real transitions after the cue: event-end contracts but remains usable; the observable-state tick loses probe transfer; the zero-input motor tick then drops teaching margin below the floor.

| Phase | 8-cue D1 pass | Train \(\gamma\) | Probe \(\gamma\) | Between L2 min | \(C\) mean from previous |
| --- | --- | --- | --- | --- | --- |
| P0 cue | 4/4 + twins | 0.144 | 0.117 | 0.688 | — |
| P1 event-end | 4/4 + twins | 0.040 | 0.026 | 0.184 | 0.326 |
| P2 observable | 0/4 (twins still pass) | 0.0102 | −0.001 | 0.043 | 0.458 |
| P3 pre-motor | identical to P2 | 0.0102 | −0.001 | 0.043 | 1.0 |
| P4 post-motor | 0/4 | 0.005 | −0.014 | 0.015 | 1.391 mean / 0.318 min |
| P5 \(\rho_{\mathrm{elig}}\) | identical to P4 | 0.005 | −0.014 | 0.015 | 1.0 |

Four-cue probes already fail at P2 (0/4). Two-cue remains D1-pass through P3; P4/P5 two-cue is marginal (perturbation-limited).

Repeat L2 at the same cue exceeds the COLLISIONMAP-style `within_l2_stable_max=0.05` at every phase, including P0 (`within_l2_max\approx 0.19`). That is reported. It is not the D1 pass rule. D1 still transfers at P0 and P1.

## What this does not authorize

This package does not install a trace, declare v31, open SCORE, raise n, or reopen lineage. Eligibility cannot reconstruct P2–P5 after collapse. A later freeze may capture **P1** (the last robust phase) and feed both ACT scoring and credit. That freeze is not this commit.

Do not retry an optimizer, competitive rule, or prototype variant. D3/D4 remain closed with the write-geometry branch.
