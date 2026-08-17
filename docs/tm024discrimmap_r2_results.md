# TM.0.24.DISCRIMMAP.R2 DEV

Decision: **robust_linear_boundary_absent**. D1 robust: **False**. D3 robust: **False**. D4 robust: **False**. D1 train clean: **False**.

Next: `robust_linear_boundary_absent`. Historical DISCRIMMAP not rescored. SCORE unopened. No neural candidate. 1536 eligibility budget stays closed. Product **0.0.004**. `earned_next=false`.

## What was measured

280 cells on unused `TM024.DISCRIMMAP.R2.DEV.` / `TWIN.`. Fit on teaching select-tick rows only. Evaluate on probes, renamed twins, and frozen perturbations. Train and probe geometric margins are separate. \(\|w\|\) excludes the intercept.

D1 is exact hard-margin SVM (SV-subset KKT). Status is `optimal` or `infeasible`. There is no soft-margin fallback.

## Result

Eight-cue D1 is **hard-margin feasible** (`optimal`, 16/16) — the first DISCRIMMAP projected-gradient dual had treated this as an ordinary train fail. Feasible is not a pass. Required 8-cue training is not clean across addresses: E1/EΔ reach \(\gamma_{\mathrm{train}}\ge 0.01\); E0/Eλ do not (0.005 and 0.007). Probe ranking fails except E1, where probe \(\gamma\approx 0.005\) and perturbation fails. D1 8-cue full pass is **0/16**. D3 and D4 also fail every 8-cue rank cell.

Two-cue can pass some D1 cells (9/16 rank, 4/8 twin). That is not robust eight-cue separability and is not `linear_interpolation_only` (training is not clean on all required 8-cue cells).

Do not install D1–D4. Do not declare v31/v32. Do not open SCORE.
