# CORTEX v30 architecture amendment

Authorized by [`lineage_collisionmap.decision.lock`](lineage_collisionmap.decision.lock) and [`lineage_persistgate.prereg.lock`](lineage_persistgate.prereg.lock).

General organism mechanism: **scalar persistence on the zero-input motor tick**. Not an L0-specific function. Frozen LINEAGE, WALLMAP, REACH, PLASTICITYMAP, ACTORCREDIT, STATEMAP, COLLISIONMAP, PERSISTGATE, and candidate v29 remain historical. Do not increase n. Do not move τ or δ. QUAL/EVAL stay sealed. FULLDEV.R7 stays sealed.

Authorized neural law (implement only after this apparatus is on `origin/main`):

1. On motor-loop ticks only (`_sensory_tick(..., record_sensory=False)`), after the existing tanh map `f`, mix \(\rho_{t+1}=p\rho_t+(1-p)\tilde\rho_{t+1}\).
2. `p` is one scalar. Select the smallest value from the preregistered grid on DEV worlds, then freeze it before scored worlds.
3. `p=0` must recover v29. Do not add a second L2 normalization that would break that identity.
4. Sensory ticks (start, cue, event-end, observable state) stay exactly v29.
5. Keep v29 action-owned delayed credit. The credited state remains the actual post-motor `ρ_elig`.
6. Do not credit a saved earlier cue ρ while the live `W_op` / `W_act_query` readout uses the collapsed motor state.
7. Do not add an extra memory vector, semantic cue/action channels, handle meaning, reward leakage, or n>64.
8. Do not edit `cortex_develop_scorers.py`. Do not rewrite historical locks. Do not reveal QUAL/EVAL.

Narrow claim: generic motor-tick persistence. Re-earn nine sanity and C4/C5/C6. Then score frozen P0–P6 on unused worlds. Lineage stays closed even if those gates pass (S10/S11 still unrepaired). Not 0.0.005.
