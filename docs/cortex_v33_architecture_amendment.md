# CORTEX v33 architecture amendment

Authorized by [`lineage_twoscale.compat.lock`](lineage_twoscale.compat.lock) and [`cortex_v32.erratum.lock`](cortex_v32.erratum.lock).

TWOSCALE v32 (post-erratum): P1 and separator geometry are sufficient; four-cue acquire passes; eight-cue acquire is **7/8** under positive-only outer products; more replay of the same update schedule cannot repair the last cue. `architectural_wall_acquire` stands on historical TWOSCALE evidence. This amendment changes **only ACT write geometry** at P1. Not another solver MAP. Not v31 W1. Not A3 bias. Not low-margin awake learning.

Preserve the entire v32 organism: fast episodic P1 memory, contradictory replacement, error-gated awake credit, REST replay of every stored episode, low-margin REST strengthen pass, slow `W_act_query` consolidation, n=64, query scoring, erratum tie band, missing-`rho_p1` fallback, checkpoint counters.

Authorized neural law (implement after this freeze):

1. Keep v32 points 1–4 unchanged (P1 capture, query scoring, eight-slot episodes, contradictory replacement).
2. **Awake trigger frozen:** update `W_act_query` only when `_act_ranking_error(p1, handle, adv)` — positive when winner ≠ handle; negative when harmful handle wins or there is no unique winner. Do **not** add margin-based awake updates.
3. **Positive awake/REST vector:** `ΔW = η·adv·(m_h − m_R) ⊗ P̂₁` where `R = { r ≠ h : score(r) ≥ max_{j≠h} score(j) − TIE_EPS }` and `m_R = (1/|R|) Σ_{r∈R} m_r` (zero if R empty). No dictionary-order rival tiebreak.
4. **Negative vector:** `ΔW = η·adv·m_h ⊗ P̂₁` only — do not depress an arbitrary alternative.
5. **REST schedule frozen:** one base replay update per valid stored episode each pass; second strengthen update when `adv>0`, unique winner equals handle, and normalized geometric margin < 0.01 — vector only, not trigger.
6. Awake `W_act_query` updates clip but do not mix into slow weights. Other plastic matrices unchanged.
7. Preserve n=64. Zero innate cue-action maps. No learned actuator intercept.
8. Do not edit `cortex_develop_scorers.py`. Do not rewrite historical locks. Do not reveal QUAL/EVAL. Do not stamp `earned_next` or 0.0.005.

Narrow claim: geometry-only competitive plasticity at P1. Score on unused `TM026.COMPETITIVE.*` worlds. Lineage stays closed even if battery passes. Not 0.0.005.
