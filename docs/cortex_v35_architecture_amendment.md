# CORTEX v35 architecture amendment

Authorized by [`lineage_gatedrehearsal.r2.decision.lock`](lineage_gatedrehearsal.r2.decision.lock).

v34/R2 confirmed the **stored-training / live-query distribution gap**: awake credit captures live P1; gated rehearsal trains stored episodes; ACT scores live P1; stored rows converge while live probes fail after REST.

v35 adds **complementary episodic–cortical recall at ACT only**:

1. **Write/replacement unchanged:** eight content-addressed P1 episodes; L2 radius **0.05**; contradictory replacement.
2. **Recall separate rule:** unique **nearest valid episode** (no distance threshold); ELIG_EPS tie at minimum → cortical fallback; pattern-complete to stored unit P1; score via existing `W_act_query`. **Never return stored handle.**
3. **Learning path frozen:** v34 awake credit, gated rehearsal, REST consolidation, competitive geometry unchanged — all on live/stored training paths only.
4. **Genome flag:** `episodic_act_recall` default **False** (v34/R2 behavior); TM028 treatment enables **True**.
5. **Canonical motor API:** `actuator_decision_scores(live_p1)` → scores, scoring_address, recall_meta — used in motor loop and all probes.
6. **Perturbation:** perturb live P1 → retrieve → complete → score (not perturb completed address).
7. **Tie behavior:** preserve `_choose_actuator` RNG tie-break; no HOLD conversion.
8. **Matched ablation:** train once, checkpoint, clone ON vs OFF (identical weights/episodes/RNG).

**If v35 fails:** report `episodic_overgeneralization` on novel-cue cells; do not add tuned recall radius.

Not another plasticity knob. Not cue names. Not larger n. Not auto `cortex.candidate.v35.lock`.
