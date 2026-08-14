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

## v6 use-skill on the plain prior

Checkpoint: **`prior_plain.pt` only**. No NOTE-copy, no lord/love in pretrain. Same collect pipeline as v5.

Three predeclared retrieve arms:

| Arm | Skill | Expected |
|-----|--------|----------|
| tool | match heading, next byte from body, logit bias **+3.0** (v0 store magnitude) | Store-works if machinery can use S |
| fewshot | frozen `NOTE:` demos without probe facts + selected note | Fail unless this LSTM in-context-copies |
| note | untaught `NOTE:` prepend | Fail (v2/v5) |

Do not train NOTE-copy to rescue fewshot/note. Do not raise the tool bias after seeing P(`v`). Classify on commit + unmount W.

## v7 native tags (no English prior)

Same key/door world as v0. Observations = bit vectors. S/W = integer `.tag` files (`door=0`, `action=2`). Genome = frozen cortex seed + rules, **not** ACGT.

Select by tag match. Collect commit/peek/off as v5. Dump-all is a control (clutter `action=1` should prefer `open`).

Pass Store-works: A correct after ρ reset; reload from `.tag` files correct; B and disable-S fail after reset; collect+unmount correct; peek+unmount incorrect; no English prose in files; weights unchanged.

## v8 boxed use-policy (cortex frozen)

Tiny linear policy may change. Cortex must not. Features = `{s_hit, w_hit}` only (no door id, no novelty). Collect is ignore/peek/commit; apply is a gate. Frozen `_apply_record_bias` still reads `action=` from the file.

Training: two-step red episodes — (1) W on, commit; (2) unmount W, reset ρ, apply from S. Reward on step 2.

Held-out: green door, `d2.tag` `{door:2, action:0}`, probe WAIT. Never in training W.

| ID | If |
|----|----|
| Confound | Cortex hash moves, or disable-S (W unmounted) still `use_key` |
| Fail | Policy hash unchanged; red unmount wrong; green unmount wrong |
| Store-works | Cortex unchanged, policy changed, red unmount `use_key`, empty S `open`, green unmount `wait` |

Do not bake `use_key` into the policy. Do not raise store bias to rescue the plot. disable-S must unmount W (peek of W is not a fact leak test).

## v9 write from a life (W has no answer)

Collect is **off**. W is clutter only (no `d0.tag`, no `d2.tag`). Boxed policy learns **when** to author a note. Frozen WHAT: on door-opening, write `{door: here, action: the act}`. Features = `{s_hit, opportunity}` (no door id).

Training: forced red curriculum (OPEN, PICK_KEY, USE_KEY) so an opening fires; write gated by policy; unmount W; reset ρ; reward on probe.

Held-out: green life (OPEN, WAIT). Same policy, no green file in W.

| ID | If |
|----|----|
| Confound | Cortex hash moves; disable-S still `use_key`; or answer file was in W |
| Fail | Policy unchanged; no authored file; red dies after ρ reset; green life fails |
| Store-works | Cortex unchanged, policy changed, S authored from events, red `use_key` and green `wait` after ρ reset, empty S / disable-S fail |

Do not copy a labeled tag from W. Do not let the policy emit `use_key` / `wait`.

