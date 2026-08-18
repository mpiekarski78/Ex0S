# CORTEX v39 architecture amendment

Authorized by [`lineage_updategeom.decision.lock`](lineage_updategeom.decision.lock). TM031–TM036 historical locks are **not** edited. TM036 is **not** rerun. v38 stays closed and dormant (`set_act_rehearse_arm` is not this law).

TM036 showed: the post-write snapshot already has a nearby jointly feasible \(W^*\) (\(\|W-W_0\|_F\approx 0.028\), \(\gamma^*\approx 0.052\)). Sequential GEM-like protection blocks movement toward that solution. v39 is **bounded cyclic projection onto the joint episodic constraint set**, not more replay, not a fitted \(\eta\), not GEM.

## Convexity proof (required before PA/Dykstra)

Motors \(v_h\) are unit. Query scores use \(q=W p_1\) and cosine \(v_h\cdot q/(|q||v_h|)\).

**Ranking is a halfspace in \(W\).** For unit motors,

\[
\mathrm{score}(h)>\mathrm{score}(r)\iff (v_h-v_r)\cdot q>0\iff \langle W,\mathrm{outer}(d,p_1)\rangle_F>0,
\]

with \(d=v_h-v_r\). The common \(|q|\) cancels. This set is convex (an open halfspace).

**Geometric \(\gamma\) is not a halfspace.** The stored-row floor is

\[
\gamma=\frac{(W^\top d)\cdot p_1}{\|W^\top d\|}\ge\tau,\qquad\tau=\texttt{ACT\_MARGIN\_FLOOR}=0.01.
\]

\(\gamma(\lambda W)=\gamma(W)\) for \(\lambda>0\). Equivalently \(d^\top W p_1\ge\tau\|W^\top d\|\), a second-order cone. Scale invariance plus a positive floor is a spherical cone, not a linear halfspace. Ordinary Crammer PA-I / Dykstra on halfspaces **cannot** be claimed as exact projection onto \(\{\gamma\ge\tau\}\) without a SOC projector.

**Authorized projector.** At the start of each cycle freeze \(W_{\mathrm{ref}}\) and set \(b=\tau\|W_{\mathrm{ref}}^\top d\|\) for every \((\text{episode},\text{rival})\) pair with \(\mathrm{adv}>0\). Each step is the exact Euclidean (Frobenius) projection onto the supporting halfspace \(\langle W,\mathrm{outer}(d,x)\rangle_F\ge b\). After the cycle, recompute \(b\). **Stop on the existing nonlinear** `_episode_rehearsal_violation` (ranking unique-winner **and** \(\gamma\ge\tau\)). Document this as linearized supporting-halfspace PA, not as “geometric \(\gamma\) is a halfspace.”

Crammer PA-I on \(W\in\mathbb{R}^{d_{\mathrm{sym}}\times n}\): \(A=\mathrm{outer}(d,x)\), \(\langle W,A\rangle=d^\top W x\), \(\|A\|_F^2=\|d\|^2\|x\|^2\). If \(\langle W,A\rangle<b\),

\[
W\leftarrow W+\frac{b-\langle W,A\rangle}{\|A\|_F^2}A.
\]

No fitted learning rate: the step is analytic.

## Controller (instance flag, not `GenomeConfig`)

Default **off** = frozen v37 credit (`write` + ranking-error one-shot + v37 Gauss–Seidel burst). Not a `GenomeConfig` field (TM031 `to_dict`).

When `pa_cyclic` or `dykstra` is bound:

1. Write the episode (unchanged lifecycle).
2. Skip fixed-\(\eta\) one-shot and the v37 burst.
3. Cycle through **all** valid \((\text{episode},\text{rival})\) supporting halfspaces, not only current violators.
4. Stop at zero violations or **16** cycles (`EPISODE_REPLAY_EPOCHS`).
5. Clip `mix_slow=False` **once per cycle**, not after every constraint.

`dykstra` additionally keeps per-constraint corrections \(I_i\): \(y=W-I_i\), \(W=\mathrm{proj}_{C_i}(y)\), \(I_i\leftarrow W-y\). Key by `slot|handle|rival`. Reset a slot’s corrections when that episode is replaced. Checkpoint \(I_i\) and `act_proj_arm`. Missing keys load as empty / `off`.

Episode handles participate **only** in consolidation constraints. They never enter ACT routing.

REST `_replay_episodes` stays the frozen v37 gated path. TM036’s failure is awake credit, not REST.

## Oracle

TM036 `closest_feasible_W` remains a **runner-only diagnostic ceiling**. Never copy \(W^*\) into `neural_cortex`. Never bind `oracle` as `act_proj_arm`.

## Hard constraints

Keep 16-pass budget. Stop on the existing violation predicate. No fitted \(\eta\). Do not add `early_raw_half_spacing` to `ACT_RECALL_MODES`. Do not modify `R` or write L2 0.05. Do not fit 44. Do not write `cortex.candidate.v39.lock` from TM037 DEV. Product **0.0.004**.
