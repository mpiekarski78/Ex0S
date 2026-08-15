# TM.0.9.2 results: antecedent MATCH / motor bar

**Date:** 15 August 2026  
**Classification:** **A Store-works** · **B Fail**  
**Run:** `runs/2026-08-15_115848_tm092`

Recipe jump: **antecedent MATCH.** A stored `X→action` may steer only when `X` is in the current observation. Genome exposes `bind_present_in_current_stream` (bool). No token ids, no `push`/`flim` list, no new policy features (`n_feat` stays 2). Cortex frozen (`a485b26b…`). `n_train` was not raised. Historical BOX was not rewritten.

## Question

If S holds `X→PRESS` and `Y→TUNE`, does the current stream decide which relation applies — or does any accessible bind→did note copy `did`?

## Headline

| Check | A MATCH | B motor bar |
|-------|---------|-------------|
| Classification | **Store-works** | **Fail** |
| Same-S cue `X` / cue `Y` | matching motors | — |
| Cue `X` + only `Y→M2` | **HOLD** | — |
| Cue `Y` + only `X→M1` | **HOLD** | — |
| Empty S + cue | **HOLD** | — |
| Permuted seeds | **3/3** Pass | — |
| Train S n files | — | **0** |

Nonce identities, motors, and filenames were permuted across seeds 12345–12347. Force-use gate also Pass (the Boolean is genome machinery, not a trained token dictionary).

## Compare

**A** is the jump: same S, cue changes, selected motor changes; relation fixed, cue does not match, HOLD. That is applicability, not another retrieval heuristic.

**B** no-cue English probes HOLD. MATCH requires an antecedent in the current stream; the old motor-bar probe does not present one. n=0. Cap not raised. Not retuned.

## Honest limits

- MATCH is a genome gate on a Boolean, not a new SGD feature. `policy.py` is unchanged.
- Historical BOX rerun after 0.9.2 (`runs/2026-08-15_115954_tm091box`): leakage **not observed** 3/3; donor **Pass** 3/3; neutrals still copy (**Fail** 3/3); transfer **Pass** 2/2 evaluable; W3 acquisition 2/3. Compatible label still **Control Fail**. MATCH stayed off. Do not require this battery to turn green.
- EVIDENCE landed in TM.0.9.3. Do not fix the no-cue English B bar.

## Reproduce

```bash
python tests/test_tm092.py
python tests/test_tm091box.py
python tests/test_tm091.py
python -m experiments.run_tm092
python -m experiments.run_tm091box --seeds 12345 12346 12347 --workers 3
```
