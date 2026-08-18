# TM.0.37.JOINTPROJ results

Cyclic supporting-halfspace projection after TM036. Product **0.0.004**.

## Decision: `jointproj_oracle_only`

Linearized Passive-Aggressive and Dykstra do not restore eight-cue acquire. Oracle \(W^*\) remains the diagnostic ceiling and is **not** installed. No `cortex.candidate.v39.lock`. 16-pass cap unchanged. 44 not installed.

- worlds: 8 diagnostic 2 v37-already 6
- routes: `['v37_already_converged', 'v37_already_converged', 'oracle_only', 'oracle_only', 'v37_already_converged', 'v37_already_converged', 'v37_already_converged', 'v37_already_converged']`
- diagnostic routes: `['oracle_only', 'oracle_only']`

Diagnostic `reg1` (both orders):

- v37 leftover 0/4/6, live 6/8, 122 updates
- PA cyclic leftover slot 6, live 7/8, 47 projections, \(\|W-W_0\|_F\approx 0.012\)
- Dykstra leftover slot 6, live 7/8, 61 projections, \(\|W-W_0\|_F\approx 0.012\)
- oracle feasible \(\gamma^*\approx 0.052\), \(\|W-W_0\|_F\approx 0.030\), live 8/8

The linearized supporting-halfspace cycle moves, and it reduces damage relative to native v37, but it does not reach the joint SOC intersection that \(W^*\) occupies. Default organism law stays frozen v37.
