# v5 results: collect from unread library W

**Date:** 14 August 2026  
**Question:** can the agent **take** a matching file from available data W, and does that become long-term only if it is **committed** to S?  
**W is not memory.** Unmounting W after a peek must return to the empty prior.

| Arm | Prior | Commit, then unmount W | Peek, then unmount W | Class (on commit) |
|-----|--------|------------------------|----------------------|-------------------|
| note | `prior.pt` | **0.988** | **0.027** (prior) | **Store-works** |
| raw | `prior_plain.pt` | **0.093** | 0.084 (prior) | **Fail** |

Runs: `runs/2026-08-13_231336_v5_note`, `runs/2026-08-13_231336_v5_raw`

## Setup

W = same 13 `.md` files as v4 (unread library). S starts empty.

Frozen collect rule: if S misses and W has a heading match, take **only that file**.

| Mode | What happens |
|------|----------------|
| **commit** | copy matching W file into S; W stays put; then unmount W and reset ρ |
| **peek** | use W for this probe only; write nothing to S; then unmount W |
| **off** | W is visible; collect disabled |

Predeclared: only **commit** may be Store-works. Peek after unmount must be prior. Collect off must ignore W.

## Headline (NOTE prior)

| Check | P(`v`) | S files |
|-------|--------|---------|
| Empty prior | 0.027 | none |
| Commit, W still mounted | 0.988 | `my-lo.md` only |
| **Commit, W unmounted, ρ empty** | **0.988** | `my-lo.md` |
| Peek, W mounted | 0.988 | **none** |
| Peek, W unmounted | **0.027** | none |
| Collect off, W mounted | 0.027 | none |
| Delete S after commit | 0.027 | none |
| W still has `my-lo.md` after commit | yes (copy, not move) | |

Weights unchanged. Collect did not ingest the other 12 library files.

Raw commit copies the same `my-lo.md` into S; P(`v`) stays ~0.093. The file was taken; this tiny LM still does not use raw replay. **Fail**.

## What this means

**Available ≠ known.** Peeking at W is a session use of the library. Long-term memory is the copy in S. That is the split: W grows without bound; S should grow only by what the drives commit.

The collect rule is still hardcoded (S miss → longest W match). Not learn-to-learn. Not RAG.

## Honest limits

- W is a folder of 13 notes, not the internet.
- Commit copies the whole matching file. No chunking.
- A model without a use-protocol (raw) can collect a correct file and still fail to *understand* it.

## Reproduce

```bash
python tests/test_library.py
python -m experiments.run_v5 --both
```
