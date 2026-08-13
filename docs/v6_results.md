# v6 results: use-skill on the plain prior (no NOTE in weights)

**Date:** 14 August 2026  
**Checkpoint:** `checkpoints/prior_plain.pt` only (NOTE-follow acc 0.025). Do not use `prior.pt`.  
**Question:** can a frozen **use-skill** read a committed `.md` after ρ reset, without putting `my love` or `NOTE:` copy into the LM weights?

| Arm | What the skill does | P(`v`) after commit + unmount W | Class |
|-----|---------------------|----------------------------------|-------|
| **tool** | heading/body → next-byte logit bias (+3.0, same size as v0 store) | **0.649** | **Store-works** |
| fewshot | frozen `NOTE:` demos (no love/lord) + selected fact | 0.053 | **Fail** |
| note | prepend `NOTE: my lo -> v` (untaught) | 0.053 | **Fail** |

Runs: `runs/2026-08-13_233601_v6_tool`, `_fewshot`, `_note`

## Setup

Same W library as v5 (13 files). Collect **commit** one match into S, unmount W, empty ρ. Weights frozen. Tool bias predeclared at **3.0** (v0 `USE_KEY` store bias), not tuned on this probe.

Fewshot demos (machinery, not facts):

```text
NOTE: qq qq -> z
NOTE: aa aa -> b
NOTE: xx xx -> w
```

No `lord` / `love` / `my lo` in those lines.

## Headline (plain prior, empty P(`v`)=0.084)

| Check | tool | fewshot | note |
|-------|------|---------|------|
| LM context after commit | **`my lo` only** | demos + `NOTE: my lo -> v\nmy lo` | `NOTE: my lo -> v\nmy lo` |
| `love` in LM context | **no** | yes (in the fact line) | yes |
| P(v) commit + unmount W | **0.649** (argmax `v`) | 0.053 | 0.053 |
| Peek then unmount | 0.084 | ~prior | ~prior |
| Collect off | 0.084 | 0.084 | 0.084 |
| Delete S | 0.084 | 0.084 | 0.084 |
| S files after commit | `my-lo.md` | `my-lo.md` | `my-lo.md` |
| Weights unchanged | yes | yes | yes |

## What this means

The missing piece was **how to read**, not a bigger wiki and not NOTE-copy in the cortex.

- **Tool:** the file holds `my lo → v`. The grammar (match heading, take next byte, bias logits) is innate machinery. The LSTM never sees the word `love`. After ρ reset and after the library is gone, P(`v`) stays up. Disable-S / delete S / peek-unmount return to prior. That is Store-works on the **plain** prior.
- **Fewshot / note:** showing the protocol in the prompt is not enough for this tiny LSTM. It does not acquire “copy the letter after `->`” in context. Same Fail as v2/v5 raw.

v1 Store-works taught the protocol **into weights**. v6 tool keeps the protocol **out of weights** and still works. That is the inside/outside split: skill in machinery, fact in the file.

## Honest limits

- Tool retrieve is **tag→byte**, like v0 tag→action. The LM is not parsing English. Do not call this reading comprehension.
- Bias +3.0 was taken from v0, not searched to make a pretty plot. P(`v`)=0.65 is not 0.99; the nudge is finite.
- Fewshot Fail means this model cannot *learn* the use-skill from three examples in the window. Learn-to-learn is still not shown.
- Do not sneak `prior.pt` NOTE-copy back in to rescue fewshot.

## Reproduce

```bash
python tests/test_v6.py
python -m experiments.run_v6 --all-modes
```
