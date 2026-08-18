# TM.0.39.JOINTSOCP contract

**Lab:** TM.0.39.JOINTSOCP · **Architecture freeze:** numerical joint SOCP consolidation. **Not a v40 candidate.**

Fresh `TM039.JOINTSOCP.DEV./TWIN.` worlds. Same post-write snapshot as TM036–TM038. Do **not** edit TM031–TM038 locks. Do **not** install TM036 `closest_feasible_W` as organism law. Do **not** write `cortex.candidate.v40.lock` from this lab.

## Why this wall

TM038 mixed routes: no cyclic projector is licensed across geometries. The only uniform mechanism supported on those diagnostic worlds is a **joint** feasible point. This lab freezes that point as a numerical SOCP, not as an “exact projector.”

## Optimization problem (frozen)

\[
\min_W \tfrac12 \|W-W_0\|_F^2
\quad\text{s.t.}\quad
d^\top W x \ge \tau \|W^\top d\|
\]

for every valid episode/rival with \(\mathrm{adv}>0\). \(\tau=\) `ACT_MARGIN_FLOOR`. Solver: **CVXPY 1.7.3 + Clarabel 0.11.1**, float64, `max_iter=200` (Clarabel default), `max_threads=1`, `direct_solve_method=qdldl`, gap/feas tols \(10^{-8}\) (Clarabel defaults). Deterministic pins, not DEV-fitted.

This is **numerical joint SOCP consolidation**. It is not local PA/Dykstra and not TM036’s two-handle closed form.

## Authoritative validation

The existing nonlinear `_episode_rehearsal_violation` remains the gate after solving. If the solver returns a non-optimal status, a tie / zero-normal (\(\|W^\top d\|\le\) `PROTO_EPS`), a negative SOC slack, a nonfinite \(W\), infeasibility, or any remaining organism violation after one `mix_slow=False` clip: **reject the entire candidate**. Restore \(W_0\). Never partial-install. Never nudge.

Record: solver status, iterations, objective, primal min-slack, \(\|W-W_0\|_F\), before/after SHA-256 of float64 \(W\), apply-or-reject.

## Credit law (instance flag, not `GenomeConfig`)

Default **off** = frozen v37. Invoke SOCP only as specified by the arm:

- `v37` — write + ranking-error one-shot + v37 burst. Never SOCP.
- `fallback_joint` — frozen v37 first; SOCP **only if** stored violations remain.
- `always_joint` — write, then SOCP from post-write \(W_0\) (no v37 plasticity). Tests whether fallback merely saves compute or changes the installed \(W\).

Apply a feasible candidate **atomically**. Episode handles stay out of ACT routing. REST stays frozen v37 gated. v38/v39 controllers stay dormant.

## Philosophical boundary

This is **global optimization during consolidation**. If Ex0S permits that as a slow-cortical mechanism, the law is legitimate. If the organism must learn through biologically local updates only, joint SOCP is outside the architecture — **stop the lineage** rather than disguise the solver as neural plasticity.

## v40 candidate gate (not this lab)

Installing the solver does **not** earn `cortex.candidate.v40.lock`. Acquire on this snapshot is not enough. A later complete causal battery must pass on untouched worlds: reversal, perturbation, history, novelty, 8×4 scaling, and later learning. Until that battery passes, no candidate lock.

## Routes (first-match on diagnostic worlds)

Diagnostic = `v37` fails the fix gate.

1. `jointsocp_no_v37_fail`
2. `jointsocp_fallback_sufficient` — fallback fixes every diagnostic world (always_joint may also)
3. `jointsocp_fallback_blocks` — only always_joint fixes; v37 pre-step changes the basin
4. `jointsocp_solver_fail` — neither joint arm fixes
5. `jointsocp_mixed_routes`

Product **0.0.004**.
