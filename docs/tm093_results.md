# TM.0.9.3 results: EVIDENCE / motor bar

**Date:** 15 August 2026  
**Classification:** **A Store-works** · **B Fail**  
**Run:** `runs/2026-08-15_125734_tm093`

Recipe jump: **EVIDENCE.** Among relations MATCH has already allowed, prefer the better-supported one. Equal evidence stays unresolved. Counts are earned in S from outcomes. Cortex frozen. `policy.py` unchanged. `n_train` not raised. Historical BOX not rewritten.

## Question

Given cue `X` and two matching hypotheses `X→M1` / `X→M2`, plus a high-support irrelevant `Y→HOLD`, can experience decide which applicable relation to trust — without filename, order, or token identity picking a winner when the evidence is equal?

## Headline

| Check | A EVIDENCE | B motor bar |
|-------|------------|-------------|
| Classification | **Store-works** | **Fail** |
| Unequal (earned 2–0 vs 0–1) | stronger `X` steers | — |
| Equal (1–0 vs 1–0) | **HOLD** / unresolved | — |
| Counterfactual S swap | preference follows S | — |
| ρ reset / wipe S | kept / **HOLD** | — |
| Y support=100 | dropped by MATCH | — |
| Permuted seeds | **3/3** | — |
| BOX-MATCH (0.9.2 make) | **Store-works** | — |
| Train S n files | — | **0** |

Lives that earned the unequal counts: M1 succeeds, M2 fails, M1 succeeds. Both X notes survived. The no-cue English bar is not this slice's pass criterion.

## Compare

**A** is the jump: MATCH removes Y; EVIDENCE ranks only applicable X relations; swap of inspectable counts (same P, same cue) flips the motor; a tie does not elect a filename.

**B** no-cue English probes HOLD. Not retuned. Do not weaken MATCH or EVIDENCE to green that bar.

## Honest limits

- Evidence is `support`/`contradiction` copied from earned `wins`/`losses`. No confidence calculus. No `+` in cortex.
- Historical BOX rerun after 0.9.3 (`runs/2026-08-15_125856_tm091box`): leakage **not observed** 3/3; donor **Pass** 3/3; neutrals still copy (**Fail** 3/3); transfer **Pass** 2/2 evaluable; W3 2/3. Compatible label still **Control Fail**. BOX-MATCH already held on the 0.9.2 make inside the 0.9.3 run.
- REVISION landed in TM.0.9.4 on this same comparison. Do not treat historical BOX neutrals as a current defect.

## Reproduce

```bash
python tests/test_tm093.py
python tests/test_tm092.py
python tests/test_tm091box.py
python -m experiments.run_tm093
python -m experiments.run_tm091box --seeds 12345 12346 12347 --workers 3
```
