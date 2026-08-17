# CORTEX v29 architecture amendment

Authorized only by [`lineage_plasticitymap.decision.lock`](lineage_plasticitymap.decision.lock).

General organism mechanism: **action-owned delayed credit**. Not an L0-specific function. Frozen LINEAGE, WALLMAP, REACH, PLASTICITYMAP, and candidate v28 remain historical. Do not increase n. Do not move τ or δ. QUAL/EVAL stay sealed. FULLDEV.R7 stays sealed.

Authorized neural law (implement only after this apparatus is on `origin/main`):

1. When the organism completes an action-selection tick, form one transient pending-action trace containing only: selection tick, operation eligibility, motor eligibility/state, selected opaque motor slot/vector, pre-action body, interaction/provenance token.
2. Actor traces are organism-authored. HOLD/STOP/RETRIEVE/WRITE do not own delayed body-consequence credit.
3. Host may **clamp** the sampled operation/handle after the selection tick. Clamp keeps saved eligibility and provenance; it replaces only the selected op/slot/vector. That is still organism-authored credit.
4. **Passive** imposed movement (body change with no organism selection tick / dropped actor trace) receives no actor credit.
5. On the next observation, compute advantage from physical body-state change and action cost. Credit the saved operation trace (`W_op`) and the saved selected-motor trace (`W_act_query` or emit query). Consume the pending actor trace exactly once.
6. Never credit the current sensory `ρ` merely because it follows the action. Use the saved action-tick eligibility.
7. Consolidate only tensors that received a nonzero update. Zero eligibility ⇒ no fast or slow actor update.
8. Policy stays factorized: `W_op` whether to ACT; `W_act_query` which opaque handle given ACT. A beneficial organism-authored ACT must increase later `P(ACT)` and `P(beneficial|ACT)`. Harm moves them oppositely.
9. Do not store token meaning or permanent handle identity. Do not add L0-specific circuitry, stage/domain branches, or capability-named heads.
10. Do not edit `cortex_develop_scorers.py`. Do not rewrite historical locks. Do not reveal QUAL/EVAL.

Narrow claim: action-owned delayed credit. Re-earn nine sanity and C4/C5/C6. Then score frozen A0–A11 cells on unused worlds. Then a newly committed reachability diagnostic. Not a G1+G3+G5 rescore. Not 0.0.005. Another lineage run stays closed unless that reachability passes.
