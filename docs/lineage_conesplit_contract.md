# TM.0.38.CONESPLIT contract

**Lab:** TM.0.38.CONESPLIT · **Not v40.** No neural edit. v39 stays an honest linearized failure.

Fresh `TM038.CONESPLIT.DEV./TWIN.` worlds. Same post-write snapshot as TM036/TM037. Do **not** edit TM031–TM037 locks. Do **not** change rehearsal cap, write L2, `R`, or `ACT_RECALL_MODES`. Do **not** install oracle \(W^*\).

## Why this wall

TM037 mixed two changes relative to \(W^*\): supporting-halfspace linearization, and a 16-cycle stop at movement \(\approx 0.012\) while \(W^*\) sits near \(0.030\). That does not isolate linearization, insufficient cyclic convergence, or cyclic projection versus a joint SOCP.

## Constraint geometry (frozen)

For each stored episode with \(\mathrm{adv}>0\) and each rival actuator, \(d=v_h-v_r\), \(x=\widehat{p_1}\), \(\tau=\) `ACT_MARGIN_FLOOR` \(=0.01\):

\[
d^\top W x \ge \tau \|W^\top d\|.
\]

This is a second-order cone in \(u=W^\top d\): \(u\cdot x \ge \tau\|u\|\). When \(\tau>0\) and \(\|u\|>0\), it already implies \(u\cdot x>0\), so separate ranking halfspaces are **not** added on SOC arms.

The closed cone includes the origin (\(0\ge 0\)). The organism’s `_episode_rehearsal_violation` treats \(\|u\|\le\) `PROTO_EPS` as \(\gamma=0\) and therefore a **fail**. `PROTO_EPS` is the existing organism constant used by `_act_geometric_margin`, not a DEV-fitted epsilon.

### Degenerate law \(W^\top d=0\)

Euclidean projection onto the organism-open set \(\{\gamma\ge\tau\}\setminus\{0\}\) does **not exist** at \(u=0\) (infimum 0, not attained). Projection onto the strict ranking set \(u\cdot x>0\) is likewise not a closed projection problem.

**Cyclic law:** if \(\|u\|\le\) `PROTO_EPS` before an SOC step, **skip** that constraint (leave \(W\) unchanged for it). Record `degenerate_skip`. Do not nudge along \(x\). Do not invent a new epsilon. If a non-degenerate point projects to the origin (polar cone), that **is** the closed-cone projection; record `projected_to_origin` and leave it. Joint SOCP may still be feasible because it treats all rows together.

## Exact SOC projector (one constraint)

Let \(u=W^\top d\), axis \(\hat x\), \(\alpha=\arccos\tau\). Euclidean projection onto the closed circular cone:

- already inside (\(u\cdot\hat x \ge \tau\|u\|\)): stay
- polar (\(\theta \ge \pi/2+\alpha\)): \(0\)
- otherwise: rotate in \(\mathrm{span}\{\hat x,u\}\) onto the generator at angle \(\alpha\)

Lift back by the minimal Frobenius change with \(W^\top d=u_{\mathrm{new}}\):

\[
W \leftarrow W + \mathrm{outer}\bigl(d,\, u_{\mathrm{new}}-W^\top d\bigr)/\|d\|^2.
\]

Dykstra corrections \(I_i\) are retained, matching TM037’s stronger cyclic method. Clip `mix_slow=False` at end of pass. Stop on the existing nonlinear violation predicate.

## Diagnostic convergence (not a fitted budget)

First of: zero violations; a cycle with \(n_{\mathrm{projections}}=0\); **256** cycles. 256 is a pre-registered power of two above 16, not fitted from TM037’s \(0.012/0.030\).

## Arms (matched clones of the write-only snapshot)

- `lin_dykstra_conv` — TM037 linearized supporting-halfspace Dykstra to diagnostic convergence
- `soc_16` — exact SOC Dykstra, **16** cycles
- `soc_conv` — exact SOC Dykstra to diagnostic convergence
- `oracle` — TM036 closest feasible \(W^*\) (joint ceiling). Diagnostic only.

A throwaway `lin_16` probe (frozen TM037 Dykstra @ 16) labels diagnostic worlds. It is not a scored arm.

## Routes (first-match on diagnostic worlds)

Diagnostic = `lin_16` probe fails the fix gate.

1. `conesplit_no_lin16_fail` — every world already fixes under 16-cycle linearized Dykstra
2. `conesplit_linearization_causal` — exact SOC fixes every diagnostic world in 16 cycles
3. `conesplit_budget_causal` — linearized Dykstra eventually fixes; SOC @ 16 does not
4. `conesplit_geometry_and_budget` — only exact SOC to convergence fixes
5. `conesplit_joint_socp_only` — only \(W^*\) fixes
6. `conesplit_capacity_wall` — oracle itself fails
7. `conesplit_mixed_routes` — diagnostic worlds disagree

**No v40 freeze. No `cortex.candidate.v39.lock` / `v40.lock`.** Product **0.0.004**.
