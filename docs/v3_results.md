# v3 results: markdown files as S (no RAG)

**Date:** 14 August 2026  
**Store:** folder of `.md` files. No embeddings, no vector index.  
**New test:** write during experience → **new agent, empty ρ, load the folder only**.

| Arm | Checkpoint | Retrieve | Classification | Run |
|-----|------------|----------|----------------|-----|
| note | `prior.pt` | `NOTE: pfx -> ch` | **Store-works** | `runs/2026-08-13_225526_v3_note` |
| raw | `prior_plain.pt` | snippet as ordinary text | **Trace-only** | `runs/2026-08-13_225527_v3_raw` |

## Question

> Before RAG: if S is just markdown on disk, can a **new** process with empty ρ still use the fact?

This is not a retrieval-quality contest. It asks whether “inspectable store” can mean **files you can open**, and whether reload equals in-process S.

## File format

Experience writes one `.md` per salient prefix slug. The probe uses `my-lo.md`:

```markdown
# my lo

my love
```

Heading = prefix. Body = snippet. Reload parses those strings. There is no embedding.

Overlapping 5-grams from repeated `my love\n` also write extra files (`e-my.md`, `y-lov.md`, …). Retrieve matches heading/prefix, so those extras are inspectable clutter, not the probe context.

## Headline (8× `my love`, probe `my lo`)

| Check | v3 note (reload .md) | v3 raw (reload .md) | v3 S off (either) |
|-------|----------------------|---------------------|-------------------|
| New agent, ρ empty, load folder | **yes** | **yes** | no files |
| Reloaded context | `NOTE: my lo -> v\nmy lo` | `my love\nmy lo` | `my lo` |
| NOTE in context | yes | **no** | no |
| Empty prior P(v) | 0.027 | 0.084 | same as arm |
| P(v) before ρ reset | 0.999 | 0.555 | 0.252 / 0.528 |
| P(v) after ρ reset, same process | 0.988 | 0.093 | = prior |
| **P(v) after new-agent reload** | **0.988** | **0.093** | n/a |
| JS(reload, same-process) | **0** | **0** | n/a |
| Delete `.md` then probe | → prior | → prior | n/a |
| Files when S disabled | none | none | — |
| Weights unchanged | yes | yes | yes |

Same-process after ρ reset and new-agent reload are **identical** (JS = 0). The folder is S.

## What this means

Putting S on disk does not change v1 vs v2. The taught NOTE prior still copies the heading/body into `v`. The plain prior still ignores raw replay after ρ reset.

What v3 adds: the fact is a **file**. A second agent that never saw the session, with ρ at zero, gets the same probe as the first agent after reset — if and only if it can read those files **and** the frozen LM knows how to use them.

That is the step before RAG: external notes, string match, no vectors.

## Honest limits

- Not RAG. No chunking, no embeddings, no ranker.
- Extra 5-gram `.md` files are real; do not pretend S is a single tidy note.
- A handwritten `my-lo.md` with the same shape is enough (see `tests/test_md_store.py`). Experience is not required if you already have the file.
- Do not treat note-arm Store-works as “markdown made the model smarter.” It reused v1’s copy protocol on file contents.

## Reproduce

```bash
python tests/test_md_store.py
python -m experiments.run_v3 --both
# or one arm:
python -m experiments.run_v3 --retrieve note
python -m experiments.run_v3 --retrieve raw
```
