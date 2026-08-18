# CORTEX v40 architecture amendment

Authorized by [`lineage_conesplit.decision.lock`](lineage_conesplit.decision.lock). TM031–TM038 historical locks are **not** edited. v39 linearized cyclic projection stays an honest failure and stays **dormant**.

TM038: no cyclic projector is licensed across geometries. The uniform mechanism still supported is a jointly feasible \(W\). v40 is **numerical joint SOCP consolidation**, not an exact projector, not GEM, not more replay.

## Philosophical boundary (read first)

This mechanism is a **global convex solver** invoked during consolidation. It is allowed only as an explicit slow-cortical choice. It is **not** local synaptic plasticity. If Ex0S later requires biologically local updates only, **stop this lineage**. Do not rename Clarabel as a neural update.

## Problem

\[
\min_W \tfrac12 \|W-W_0\|_F^2
\quad\text{s.t.}\quad
d^\top W x \ge \tau \|W^\top d\|
\quad \forall \text{ valid episode/rival with }\mathrm{adv}>0.
\]

Pinned solver: CVXPY **1.7.3**, Clarabel **0.11.1**, float64, `max_iter=200`, `max_threads=1`, `direct_solve_method=qdldl`, gap/feas \(10^{-8}\).

After a candidate is returned, **clip once** (`mix_slow=False`) and evaluate `_episode_rehearsal_violation`. Any solver failure, zero-normal/tie, negative slack, nonfinite \(W\), or remaining organism violation → reject entirely, keep \(W_0\). No partial install. No nudge. No DEV-fitted epsilon.

## Instance flag (not `GenomeConfig`)

Default **off** = frozen v37 credit. `fallback_joint`: v37 then SOCP iff violations remain. `always_joint`: SOCP from post-write \(W\) every credit (matched scheduling contrast). Checkpoint `act_socp_arm` only. REST unchanged. Episode handles never enter ACT routing.

## Candidate lock is not this amendment

A passing TM039 acquire cell is **not** `cortex.candidate.v40.lock`. Required later, on untouched worlds: reversal, perturbation, history, novelty, 8×4 scaling, later learning. Product **0.0.004**.
