# Three-memory experiment protocol (v0)

## Question

> Can frozen innate drives + learning rules fill an **inspectable** world-knowledge store from experience, such that facts **survive reset of the working trace** — while the trace alone does not?

## Predeclared categories

| ID | Meaning |
|----|---------|
| Fail | Store writes junk, or behavior still dies when the trace is reset |
| Trace-only | Same as BDH Category B: ρ moves the next step; reset wipes it |
| Store-works | After experience, reset ρ, facts remain via the store and can be inspected |
| Confound | Slow weights silently absorbed the facts (illegal in v0) |

## Three pieces

1. **Frozen cortex** — species prior (sensors/dynamics). SHA256 must not change.
2. **Working trace ρ** — session EMA over embeddings. Reset is a first-class test.
3. **World store S** — explicit JSON records `{what, when, drive_scores, tags}`.

Innate drives (frozen thresholds): novelty / prediction-error vs ρ, integrity-cost on failure.
Write rule: on salient failure/success events at a door, write a structured fact (tags + template `what`) to S.
Retrieve rule: matching tags bias action logits (toy: not NLP over `what`).

The environment reports **events** (`open_failed`, `key_worked`), not a pre-labeled fact string.
## v0 world

Fact: `red door opens only with key`.

- **A** experiences the contingency (OPEN fails, USE_KEY succeeds) with S on.
- **B** experiences a blue-door foil (no red-door fact).
- **disable-S** same experience as A, writes blocked.
- Probe: `probe_red_with_key` → correct action is `use_key`.

## Pass criteria (Store-works)

- A correct after ρ reset, S kept, and the fact is in `store_A.json`
- B incorrect on the same probe after ρ reset
- disable-S incorrect after ρ reset (BDH-like Category B)
- Resetting S removes the effect
- Weight hash unchanged
- Twin identical experience → ρ L2 ≈ 0

## Comparison to BDH

See [`comparison_bdh.md`](comparison_bdh.md). BDH is the trace-only baseline ([mpiekarski78/bdh](https://github.com/mpiekarski78/bdh)).

## v1 language protocol

Same categories. Frozen tiny byte LM (syntax + NOTE-copy; lord/love/`my lo` stripped). Experience is 8× `my lord` vs 8× `my love`. Probe `my lo`.

- **A/B** write 5-gram facts to S when novelty or next-byte error is high.
- Retrieve: longest stored prefix that is a suffix of the probe, injected as `NOTE: {what}\n`.
- disable-S: session prefix→byte buffer only (cleared on ρ reset).
- Pass Store-works: S-on P(v) after ρ reset ≥ prior + 0.10; S-off after reset within 0.10 of prior; inspectable `my lo -> v`; weights unchanged.

Predeclared in `experiments/run_v1.py`.

## v2 raw retrieve (no NOTE-copy)

Same probes and categories. Prior is trained on **stripped Shakespeare only** (`--plain`). Retrieve prepends the stored **snippet as ordinary text** (`my love\\n` + `my lo`), not a taught `NOTE:` format.

Pass Store-works on the same numeric thresholds. If the tiny LM cannot use raw context, classify **Fail** or **Trace-only**. Do not add NOTE training to rescue the plot.

## v3 markdown files (no RAG)

S is a **folder of `.md` files** (heading = prefix, body = snippet). No embeddings.

After experience, copy the folder and build a **new agent** with empty ρ that only reloads those files. Classify on the reloaded probe, same numeric thresholds as v1.

Two predeclared arms:

| Arm | Prior | Retrieve | Expected if v1/v2 hold |
|-----|-------|----------|------------------------|
| note | `prior.pt` | `NOTE:` | Store-works (file ≈ JSON S) |
| raw | `prior_plain.pt` | raw snippet | Trace-only (file unread by the LM) |

The new claim is **inspectable persistence on disk**, not a better retrieve. Do not add a vector index here.

## v4 select among many notes

S has 13 `.md` files. One matches `my lo`. Traps are shorter suffixes (`lo`, `my l`). Retrieve **select** = longest matching heading only. Control **dump-all** concatenates every file.

Classify on select, same P(`v`) thresholds as v1. Dump-all is reported, not labeled. Predeclare: note+select may Store-works; raw+select may Fail (no session, unused file). If dump-all matches select, say N is too small to matter.

## v5 collect from unread W

W is a second folder (available data). S starts empty. Frozen rule: S miss and W heading match → take **one** file.

| Mode | Durable after unmount W + ρ reset? |
|------|--------------------------------------|
| commit (copy W→S) | yes if the LM can use S |
| peek (session only) | **no** |
| collect off | **no** |

Classify on **commit + unmount W**. Peek after unmount must sit at prior. Do not ingest the whole library.
