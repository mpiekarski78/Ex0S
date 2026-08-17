# CORTEX v28 architecture amendment

Authorized only by [`lineage_wallmap.decision.lock`](lineage_wallmap.decision.lock) (primary bottleneck: Q4 credit chain).

Restore a general credit-path law. Do not add L0-specific circuitry. Frozen LINEAGE engine candidate and WALLMAP results remain historical. Do not increase n. Do not move τ or δ. QUAL/EVAL stay sealed. FULLDEV.R7 stays sealed.

Authorized neural law (implement only after this apparatus is on `origin/main`):

1. Three-factor and prediction updates apply only when eligibility is active (`max|ρ_elig|` above a numerical floor).
2. `_clip_and_consolidate` runs only on tensors that received a nonzero credit or prediction increment this tick.
3. No eligibility / no credit update on a tensor ⇒ no plastic motion on that tensor, including consolidation.
4. Still clear `_pending` and record `pred_err` when eligibility is inactive.
5. Do not add capability-named shortcuts, stage/domain branches, or L0-specific heads.
6. Do not edit `cortex_develop_scorers.py`. Do not rewrite LINEAGE or WALLMAP locks. Do not reveal QUAL/EVAL.

Narrow claim: credit-path causality. Re-earn nine sanity and C4/C5/C6. Then run a newly committed reachability diagnostic on unused worlds. Not a G1+G3+G5 rescore. Not 0.0.005.
