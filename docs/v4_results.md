# v4 results: select the right `.md` among many

**Date:** 14 August 2026  
**Question:** as S grows, can the agent **use** the matching note, or does it have to dump the folder?  
**Not RAG.** Longest heading suffix that matches the probe. Traps (`lo`, `my l`) are shorter on purpose.

| Arm | Prior | Select P(`v`) | Dump-all P(`v`) | Class |
|-----|--------|---------------|-----------------|-------|
| note | `prior.pt` | **0.988** | **0.007** | **Store-works** |
| raw | `prior_plain.pt` | **0.093** | 0.051 | **Fail** |

Runs: `runs/2026-08-13_231335_v4_note`, `runs/2026-08-13_231335_v4_raw`

## Setup

13 handwritten files in S. One is `# my lo` / `my love`. Twelve are clutter plus shorter suffix traps. New agent, empty ρ, no experience this session. Weights unchanged.

Select must inject **one** note and reject 12. Dump-all concatenates all 13 (NOTE lines + bodies) as a control. Dump is **not** the classification.

## Headline

| Check | v4 note, select | v4 note, dump-all | v4 raw, select |
|-------|-----------------|-------------------|----------------|
| n_files | 13 | 13 | 13 |
| Chosen / rejected | 1 / 12 | 13 / 0 | 1 / 12 |
| Context | `NOTE: my lo -> v\nmy lo` | clutter + traps + love | `my love\nmy lo` |
| Distractors in context | **no** | **yes** | no |
| Empty prior P(v) | 0.027 | 0.027 | 0.084 |
| P(v), ρ empty | **0.988** | **0.007** | 0.093 |
| Delete S | → prior | — | → prior |

Dump-all **hurts** the NOTE prior (0.007 < empty 0.027). Selection is not cosmetic. At 13 files, stuffing the window already breaks the use-protocol.

Raw select still does not move P(`v`) (same as v2/v3). Classified **Fail**, not Trace-only: there was no session residue in this run, only an unused file.

## What this means

Growth of S requires a **select rule** (here: longest matching heading). That rule is machinery, not Wikipedia. It is still string match, not embeddings.

## Honest limits

- 13 files, not a wiki. The dump collapse is real at this N; do not claim a vector index is required yet.
- Traps were designed as shorter suffixes. A second file also headed `# my lo` would be a conflict, not tested.
- Select is hardcoded. This is not learn-to-learn.

## Reproduce

```bash
python tests/test_library.py
python -m experiments.run_v4 --both
```
