# CORTEX v32 architecture amendment

Authorized by [`lineage_affinemap.r2.decision.lock`](lineage_affinemap.r2.decision.lock) and [`lineage_affinemap.r2.decision.addendum.lock`](lineage_affinemap.r2.decision.addendum.lock).

AFFINEMAP R2: homogeneous D1 ranks four- and eight-cue maps; existing PA fails four-cue acquire ranking; learned actuator bias does not uniquely help. Representation is sufficient. Online optimization is the wall. No further solver MAP.

General organism mechanism: **fast episodic P1 memory plus slow cortical consolidation**. Not an L0-specific function. Not v31 W1. Not A3 bias. Not compact PA as the success path. Frozen LINEAGE through AFFINEMAP R2, and candidates v29/v30, remain historical. v31 W1 stays uninstalled. Do not increase n. Do not move τ or δ. QUAL/EVAL stay sealed. FULLDEV.R7 stays sealed.

Authorized neural law (implement after this freeze):

1. Capture the event-end address: after the innate `v_end` sensory tick, store unit `ρ` as `last_p1`. This is PHASEMAP P1, not a 512-row temporal TRACE.
2. Score and credit ACT from `last_p1` when it is set. `W_op` / `W_pred` keep action-owned delayed credit on the live post-motor `ρ_elig`. Default `act_score_mode` remains `query`. Do not enable W1 prototypes.
3. Store up to eight content-addressed P1 episodes `(p1, handle, adv)`. No cue strings. Empty at birth. Match radius 0.05 L2 on unit P1 (frozen MEMORYLIFECYCLEMAP radius, not a new search). State budget `8 × 64 = 512` scalars. This is not the closed 512-row event-end TRACE and not 1536 eligibility.
4. On a unique content match, replace a contradictory episode (sign flip, or positive credit to a different handle). Otherwise refresh. If the store is full and there is no match, evict the oldest.
5. While awake, update `W_act_query` with the existing advantage-modulated outer product on ranking error at P1. Always write or replace the episode when ACT credit is nonzero.
6. During REST, after the existing host rest opportunity, replay stored episodes for 16 epochs (frozen MEMORYLIFECYCLEMAP budget). Re-apply the stored signed association. If the unique winner is already correct and geometric margin is below 0.01, strengthen the correct association. Then mix fast `W_act_query` into `W_slow` with existing β.
7. Awake `W_act_query` updates clip but do not mix into slow weights. Other plastic matrices keep the existing per-update β mix.
8. Preserve n=64. `W_act_query` stays zeros at birth. No innate cue-action mappings. No learned actuator intercept. No handle meaning in sensory input.
9. Do not edit `cortex_develop_scorers.py`. Do not rewrite historical locks. Do not reveal QUAL/EVAL. Do not stamp `earned_next` or 0.0.005.

Narrow claim: two-timescale ACT learning on P1 episodes. Score acquire, stability after REST, ecological reversal, and specificity on unused worlds. Lineage stays closed even if that battery passes until a later reopen. Not 0.0.005.
