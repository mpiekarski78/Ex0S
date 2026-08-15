# TM.0.9.4 results: REVISION / motor bar

**Date:** 15 August 2026  
**Classification:** **A Store-works** · **B Fail**  
**Run:** `runs/2026-08-15_143334_tm094`

Recipe jump: **none.** Same 0.9.3 comparison. Later outcomes can withdraw and reverse a preference among surviving MATCH-eligible hypotheses. Cortex frozen. `policy.py` unchanged. Genome delta vs 0.9.3: **0**. `n_train` not raised. Historical BOX not rewritten.

## Question

Given cue `X` and two stored rivals `X→M1` / `X→M2`, can experience first prefer M1, then return to unresolved, then prefer M2 — without a new relation, a new token, a recency feature, or a new search trick?

## Headline

| Check | A REVISION | B motor bar |
|-------|------------|-------------|
| Classification | **Store-works** | **Fail** |
| Walk | **HOLD → M1 → HOLD → M2** | — |
| Mid-life ρ reset, then reverse | M1 then M2 | — |
| Same final S evidence, different order | same motor | — |
| Permuted seeds | **3/3** | — |
| Genome delta vs 0.9.3 | **0** | — |
| BOX-MATCH (0.9.2 make) | **Store-works** 3/3 | — |
| Train S n files | — | historical no-cue bar |

Early revision lives: M1+, M1+, M2− → M1 (2/0 vs 0/1). Reset ρ. Continue: M1−, M1−, M2+, M2+, M2+ → M2 (2/2 vs 3/1). The exact arithmetic is not the claim. The same stored rivals reversed because later experience changed their evidence.

## Compare

**A** is the jump: EVIDENCE is not a one-way promotion. It can withdraw belief. Persistent S remains editable after ρ reset — not merely “a finished fact survives.”

**B** no-cue English probes do not PRESS. Not retuned. Do not weaken MATCH or EVIDENCE to green that bar.

## Honest limits

- No probabilities. No Bayesian vocabulary. No decay. No weighted recency. No source trust. No `evidence_score = support - contradiction` in cortex. The existing `(support, -contradiction)` comparison already reverses.
- Historical BOX neutrals remain frozen 0.9.1 behavior (no cue, MATCH off). They are not a current capability defect. Genome delta 0 means that probe cannot have changed. BOX-MATCH is the current-organism relevance control and still holds.
- Next is a fixed-genome world family, not more epistemic machinery.

## Reproduce

```bash
python tests/test_tm094.py
python tests/test_tm093.py
python tests/test_tm092.py
python tests/test_tm091box.py
python -m experiments.run_tm094
```
