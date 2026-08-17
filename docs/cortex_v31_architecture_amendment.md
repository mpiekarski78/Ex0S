# CORTEX v31 architecture amendment

Authorized by [`lineage_motorpersist.decision.lock`](lineage_motorpersist.decision.lock) and [`lineage_motorpersist.reaudit.lock`](lineage_motorpersist.reaudit.lock).

General organism mechanism: **exchangeable actuator-local prototype rows** for ACT ranking and ACT credit. Not an L0-specific function. Frozen LINEAGE, WALLMAP, REACH, PLASTICITYMAP, ACTORCREDIT, STATEMAP, COLLISIONMAP, PERSISTGATE, MOTORPERSIST, and candidates v29/v30 remain historical. Do not increase n. Do not move τ or δ. QUAL/EVAL stay sealed. FULLDEV.R7 stays sealed.

Authorized neural law (implement only after this apparatus is on `origin/main`):

1. No second global recurrent/context state. Add bounded actuator-local plastic state: one fast and one slow 64-d row per active actuator. `state_budget = 2 * H_max * 64` with `H_max = 8`.
2. Prototype rows are registry slots keyed by opaque handles, never by cues. Handle strings never enter sensory input or prototype contents.
3. Score ACT with `actuator_scores(rho)`: cosine of the live prototype with `ρ` if the row is initialized, else exactly zero. `_motor_loop` and `motor_scores` must call this method. Preserve `rng_motor` ties when every score is zero.
4. Credit only the chosen handle: `z = proto + η_act * adv * ρ̂`, then unit-normalize or zero. `adv == 0` is bitwise no update. Then apply the existing β blend and re-unit-normalize both live and slow.
5. Retain `W_act_query` for checkpoint compatibility. Do not update it during v31 ACT credit. Do not let REST/consolidation modify it as a hidden actor. `W_op` still chooses ACT versus HOLD.
6. Keep v29 action-owned delayed credit, `ρ_elig`, EMIT, opaque `bind_actuators`, v30 persist at `p=0`.
7. Do not put a 64×64 inverse-covariance / RLS in the organism. W2 is runner-only.
8. Do not add semantic cue/action channels, instincts, reward leakage, or n>64.
9. Do not edit `cortex_develop_scorers.py`. Do not rewrite historical locks. Do not reveal QUAL/EVAL.

Narrow claim: effector-local plastic write geometry. Re-earn nine sanity, A0–A11, and C4/C5/C6. Then score frozen WRITEGEOM cells on unused worlds. Lineage stays closed even if those gates pass (S10/S11 still unrepaired). Not 0.0.005.
