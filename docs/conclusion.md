# Conclusion: three-memory v0

**Date:** 14 August 2026  
**Classification:** **Store-works**  
**Code:** [mpiekarski78/three-memory](https://github.com/mpiekarski78/three-memory)  
**BDH baseline:** [mpiekarski78/bdh](https://github.com/mpiekarski78/bdh) Category B (trace-only)

This is the deliverable for the first three-memory question. Evidence, not a product.

---

## Verdict

Frozen innate drives + write/retrieve rules **can** fill an inspectable world-knowledge store from experience. After ρ is reset, the fact still steers behavior **if and only if** it lives in S. With S disabled, the same experience behaves like BDH Category B: useful in-session, gone after reset.

Do not put a life of knowledge in ρ. Put it in S.

---

## Question

> Can frozen innate drives + learning rules fill an **inspectable** world-knowledge store from experience, such that facts **survive reset of the working trace** — while the trace alone does not?

| ID | Meaning | v0 |
|----|---------|----|
| Fail | Store junk / reset still kills the fact when S is on | no |
| Trace-only | ρ moves the next step; reset wipes it | **yes when S off** |
| Store-works | After experience, reset ρ, fact remains via S and is inspectable | **yes when S on** |
| Confound | Slow weights absorbed the fact | no (SHA256 unchanged) |

---

## Setup

| Item | Value |
|------|-------|
| World | key/door: `red door opens only with key` |
| Cortex | frozen random encoder + action head (seed 1337) |
| ρ | EMA embed + last successful action (session only) |
| S | JSON fact records; retrieve biases logits |
| Drives | novelty vs ρ; integrity-cost on failure |
| Controls | A vs B, disable-S, reset S, twin ρ, weight hash, ρ restore |

## Headline results (seed 12345)

| Condition | Probe `use_key` on red door with key |
|-----------|--------------------------------------|
| A after experience, before ρ reset | correct |
| A after ρ reset, S kept | **correct** |
| B (foil life) after ρ reset | incorrect (`open`) |
| disable-S before ρ reset | correct (session residue) |
| disable-S after ρ reset | **incorrect** |
| reset S then probe | incorrect |
| Weights unchanged | true |
| Twin ρ L2 | 0 |
| Fact in `store_A.json` | `"red door opens only with key"` |

---

## Implication

| Role | Where |
|------|-------|
| Species prior (sensors/dynamics) | Frozen cortex |
| “I just did this and it worked” | ρ, discard on reset |
| Beliefs / world facts / inspectable history | **S** |

Public BDH answered: ρ alone is Category B. This sibling answers: add an explicit store and the Category B ceiling lifts for **inspectable facts**, without pretending ρ is long-term memory.

Winning v0 does not reopen Category D on BDH ρ.

---

## Limitations (honest)

- This is a **tiny designed world**, not an emergent LLM. Store→action uses **tags** (`door=red` → prefer `use_key`), not natural-language understanding of `what`.
- The `what` string is a **fixed template** written by the learning rule when events fire; the environment no longer injects a labeled lesson string.
- Fallback forced curriculum (OPEN → PICK_KEY → USE_KEY) runs only if free policy never succeeds; reported in metrics as `n_forced_steps`.
- Foil B still forces `OPEN` on the blue door so A/B experience differs by design.
- Winning Store-works does **not** reopen Category D on BDH ρ.

## Reproduce

```bash
python -m experiments.run_v0
python -m experiments.train_prior
python -m experiments.run_v1
```

Numbers: `runs/` (gitignored). Comparison: [`comparison_bdh.md`](comparison_bdh.md). Protocol: [`protocol.md`](protocol.md). v1 table: [`v1_results.md`](v1_results.md).

## v1 (language)

Same three boxes, same probes as BDH (`my lo` → r/v). Frozen tiny byte LSTM; lord/love stripped from pretrain; S retrieved as `NOTE:` context.

**Classification: Store-works.** After 8× `my love`, P(`v`) after ρ reset is **0.988** with S on, **0.027** (empty prior) with S off. Fact in JSON: `my lo -> v`. BDH published: the same probe’s association dies on ρ reset.

v1 does not reopen Category D on BDH ρ. It shows the missing box on the *language* probes is still the inspectable store **plus a taught way to read it**.

## v2 (raw retrieve)

No NOTE-copy in pretrain (NOTE-follow acc 0.025). Retrieve prepends `my love\n` as ordinary text. Classification: **Trace-only**. After ρ reset, P(`v`)≈0.093 vs prior 0.084. The fact is in S and in the prompt; this tiny LSTM does not use it. Details: [`v2_results.md`](v2_results.md).

## v3 (markdown files, no RAG)

S is a folder of `.md` files. After experience, a **new** agent with empty ρ loads only that folder. Same-process vs reload JS = 0.

- NOTE prior: **Store-works**, P(`v`)=0.988 — the file is enough because the LM was taught to copy `NOTE:`.
- Plain prior: **Trace-only**, P(`v`)=0.093 — the same file sits in the prompt unused.

Disk persistence ≠ a use-protocol. Not RAG. Details: [`v3_results.md`](v3_results.md).

## v4 (select) and v5 (collect)

v4: 13 notes. Select the matching heading (reject 12). NOTE prior **Store-works** P(`v`)=0.988. Dump-all **collapses** to 0.007. Raw **Fail**.

v5: unread library W. **Commit** copies `my-lo.md` into S; after unmounting W, P(`v`)=0.988. **Peek** works while W is mounted, then returns to prior. Collect off ignores W. Raw commit still **Fail**.

Available data is not memory. Details: [`v4_results.md`](v4_results.md), [`v5_results.md`](v5_results.md).

## v6 (use-skill, plain prior)

No NOTE-copy in the cortex. **Tool** grammar reads the committed file (heading → next byte, bias +3.0). LM window is only `my lo`. Classification: **Store-works**, P(`v`)=0.649. Peek/unmount and delete S return to prior 0.084.

Three in-context `NOTE:` demos, and untaught NOTE prepend, both **Fail** (P(`v`)≈0.053). This LSTM does not acquire the protocol from the prompt. Details: [`v6_results.md`](v6_results.md).

## v7 (native tags)

No English prior. Genome = frozen cortex seed, not DNA letters. Notes are `door=0` / `action=2`. Classification: **Store-works**. Reload files and collect-from-W both yield `use_key` after ρ reset; peek and dump-all do not. Details: [`v7_results.md`](v7_results.md).

## v8 (boxed use-policy)

The collect/apply box may change; cortex SHA256 must not. Policy features exclude door identity. Classification: **Store-works**. After training, red commit+unmount yields `use_key`; held-out green (`d2.tag`, never in train W) yields `wait`; empty S and disable-S stay `open`. The motor act still comes from the file’s `action=`. Details: [`v8_results.md`](v8_results.md).

## v9 (write from a life)

W has no answer file. The policy learns **when** to author a note from a door-opening; the frozen template is `{here, that act}`. Classification: **Store-works**. Red life writes `d0.tag` (`action=2`) and `use_key` after ρ reset; held-out green life writes `d2.tag` (`action=0`) and `wait`. Empty S and disable-S stay `open`. Cortex unchanged. Details: [`v9_results.md`](v9_results.md).

## v10 (free life)

No forced OPEN→PICK_KEY→USE_KEY. The agent explores percept-legal acts, authors the note if a door opens, then a **greedy** probe after ρ reset. Classification: **Store-works**. `n_forced=0`. Red life was `pick_key … wait … use_key`; green found `wait` without a script. Details: [`v10_results.md`](v10_results.md).

## v11 (select among authored notes)

Two free lives fill one S with `d0.tag` and `d2.tag`. Classification: **Store-works**. Select: red `use_key`, green `wait`. Dump-all: red `wait` (the other life leaks). Empty S and disable-S stay `open`. Details: [`v11_results.md`](v11_results.md).

