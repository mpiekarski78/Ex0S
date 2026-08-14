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

## v12 (learn select vs dump)

A boxed retrieve head chooses select vs dump without door identity. Classification: **Store-works**. Untrained dumps (red `wait`). Trained selects (red `use_key`, green `wait`). Held-out blue life authors `d1.tag` and greedy probe `open`. Dump-all still waits. Cortex unchanged. Details: [`v12_results.md`](v12_results.md).

## v13 (copy `action=` from the file)

A boxed use-gate learns when to copy the file’s integer into motor logits. No USE_KEY/WAIT table on this path. Classification: **Store-works**. Untrained (empty S or planted `d0.tag`) stays `open`. Trained red `use_key`. Held-out green `wait` (integer was never in the head). Dump-all still mixes. Cortex unchanged. Details: [`v13_results.md`](v13_results.md).

## v14 (pick-one vs write schema)

Two boxed heads, same frozen cortex. **A** learns to apply one matching note (newest `when=`) instead of summing every `action=`. **B** learns to put `action=` in the note instead of `{door}` only. Both **Store-works**. Untrained A mixes (`wait`); untrained B writes door-only (`open`). Trained red `use_key`; held-out green `wait`. Details: [`v14_results.md`](v14_results.md).

## v15 (joint, no clamps)

Write WHEN, schema, use-gate, and pick-one trained together. Classification: **Store-works**. Untrained conflict stays `open`. Trained red `use_key` (newest complete file). Held-out green `wait`. Apply-all still mixes. No `force_use` / `force_write`. Cortex unchanged. Details: [`v15_results.md`](v15_results.md).

## v16 (ok= vs newest, shared return)

**A** learns to prefer a success-marked note over a newer junk file. Classification: **Store-works**. Untrained recency prior `wait`. Trained red `use_key` (`ok=1`). Held-out green `wait`. Newest-wins control still `wait`. **B** uses one shared return on the v15 joint setup. Classification: **Fail** (last-50 = 0, red stays `open`). Split credit was load-bearing. Details: [`v16_results.md`](v16_results.md).

## v17 (read `do=` vs match `here=`)

**A** learns to copy `do=` instead of `action=`. **B** learns to match `here=` instead of `door=`. Both **Store-works**. Untrained planted alt-name stays `open`. Trained red `use_key`; held-out green `wait`. Controls that keep the old name fail. Cortex unchanged. Details: [`v17_results.md`](v17_results.md).

## v18 (write `do=` vs write `here=`)

**A** learns to emit `do=` instead of `action=` (read frozen to `do=`). **B** learns to emit `here=` instead of `door=` (match frozen to `here=`). Both **Store-works**. Untrained writer stays `open` (`action=` / `door=`). Trained red `use_key`; held-out green `wait`. Controls that keep the old write-name fail. Cortex unchanged. Details: [`v18_results.md`](v18_results.md).

## v19 (shared value-name vs shared place-name)

Neither side frozen. Untrained write/read disagree. **A** met on `do=` (writer moved). **B** met on `door=` (matcher moved). Both **Store-works**. Untrained `open`. Trained red `use_key`; held-out green `wait` on the same name. Write+use without agreement stays `open`. Cortex unchanged. Details: [`v19_results.md`](v19_results.md).

## v20 (find unread W vs find vs junk)

Unread library W, not authored S. Collect is the frozen v5 commit-on-hit rule. **A** finds `{here:0, action:2}` in `p99.tag`. **B** must prefer that page over `door=` junk (`wait`). Both **Store-works**. Untrained `open`. Trained red `use_key` after unmount W; held-out green `wait`. B’s `door=` control copies junk `wait`; junk-only W stays `open`. Cortex unchanged. Details: [`v20_results.md`](v20_results.md).

## v21 (first-file vs dump-all among W hits)

Many unread pages share `here=`. Collect’s filename-first `w_hits[0]` is the wrong prior. **A** learns newest `when=` over `aaa.tag` junk. **B** learns newest over dumping every match. Both **Store-works**. Untrained `wait`. Trained red `use_key` from `p99.tag` after unmount W; held-out green `wait`. Recency-swap (newest is junk) stays `wait`. Cortex unchanged. Details: [`v21_results.md`](v21_results.md).

## v22 (complete vs stub / joint no clamps)

**A** drops planted recency. A stub `{here:0}` sorts first; the useful page has `action=`. **B** runs match + newest-pick + use-gate together with no clamps. Both **Store-works**. Untrained `open`. Trained red `use_key` after unmount W; held-out green `wait`. A’s complete-is-junk and stub-only controls fail; B’s `door=` / first-file / use-off controls fail. Cortex unchanged. Details: [`v22_results.md`](v22_results.md).

## v23 (joint find+complete+use / shared return)

**A** runs match + complete-vs-stub + use together with no `when=` and no clamps, under split credit. **Store-works**. Untrained `open`. Trained red `use_key` from `p99.tag` after unmount W; held-out green `wait`. Freeze-match `door=`, freeze-stub, and use-off all fail. **B** uses one shared return on the same stack. **Fail**. Last-50 0.00; `update()` never ran; red stayed `door=` junk. Same starvation as v16 B, now on unread W. Cortex unchanged. Details: [`v23_results.md`](v23_results.md).

## TM.0.1.0 (open query names / shared return)

Query keys come from files, not `{door, here}`. **A** split and **B** shared return both **Store-works**. Untrained queries `action=` and stays `open`. Trained red `use_key` from `p99.tag` `{loc:0, action:2}` after unmount W; held-out green `wait`. Restored match menu finds nothing. Use-off stays `open`. loc-is-wait stays `wait`. Two-head shared return worked here; it does not overturn v16/v23 starvation on larger joints. Cortex unchanged. Details: [`tm010_results.md`](tm010_results.md).

## TM.0.1.1 (open copy names / shared return)

Copy keys come from the hit, not `{action, do}`. Query frozen to the files’ place key. **A** split and **B** shared return both **Store-works**. Untrained copies `loc=` and stays `open`. Trained red `use_key` from `act=2` after unmount W; held-out green `wait` (`act=0`, not place code 2). Restored copy menu misses. Use-off stays `open`. Cortex unchanged. Details: [`tm011_results.md`](tm011_results.md).

## TM.0.1.2 (messy retrieve / shared return)

Rank unread files; exact `loc=` / `door=` misses. Useful page `{where:0, action:2, pad:7}`. **A** split and **B** shared return both **Store-works**. Untrained takes clutter `p0.tag` and stays `open`. Trained red `use_key` from `p99.tag` after unmount W; held-out green `wait`. Exact match finds nothing. Use-off stays `open`. Cortex unchanged. Details: [`tm012_results.md`](tm012_results.md).

## TM.0.2.0 (scale of W / shared return)

Same search recipe; W is **256** messy files. **A** split and **B** shared return both **Store-works**. Untrained takes early clutter with the door code and stays `open`. Trained red `use_key` from `p99.tag` after unmount W; held-out green `wait`. Exact match finds nothing. Use-off stays `open`. Cortex unchanged. `w_n=256`. Details: [`tm020_results.md`](tm020_results.md).

## TM.0.3.0 (a life / shared return)

Free life find/commit (not probe→unmount→probe train). **A** **Store-works**: commits `p99.tag` in life; after ρ reset W gone → `use_key`; held-out green `wait`; `n_forced=0`. **B** **Fail**: shared return last-50 0; red stays on clutter. Cortex unchanged. Details: [`tm030_results.md`](tm030_results.md).

## TM.0.3.1 (documents / shared return)

Free life over unread `.md` documents (prose + embedded `k=v`). **A** **Store-works**: commits from `p99.md`; after ρ reset W gone → `use_key`; held-out green `wait`. **B** **Fail**: shared return last-50 0. Cortex unchanged. Not English NLP. Details: [`tm031_results.md`](tm031_results.md).

## TM.0.3.2 (prose retrieve / shared return)

Pure prose `.md` (no filed `action=` / `where=`). Digits → anonymous `n*`; vname picks the motor int. **A** **Store-works**; **B** **Fail** (shared return last-50 0). Cortex unchanged. Digit scan is not English. Details: [`tm032_results.md`](tm032_results.md).

## TM.0.4.0 (channel dial / shared return)

Left the key/door room. Same prose machinery on `ChannelDialWorld`. **A** **Store-works**: after ρ reset W gone → PRESS; held-out C TUNE (not place-copy). **B** **Fail**: shared return never commits useful pages; C wrong. Cortex frozen (5-action hash). Details: [`tm040_results.md`](tm040_results.md).

## TM.0.5.0 (no answer integers / shared return)

Unread W has no place/motor digits. Copy an innate motor name token (`press` / `tune`). **A** **Store-works**: after ρ reset W gone → PRESS; held-out C TUNE. **B** **Fail**: shared return last-50 0. Developmental rule locked. Not English NLP. Details: [`tm050_results.md`](tm050_results.md).

## TM.0.5.1 (correct a wrong commit / shared return)

Detect fail, drop clutter in S, retry, keep after ρ reset. Search frozen untrained. **A** **Store-works**: n_revised=11, then `p99` / PRESS; held-out C TUNE. Revise-off stays HOLD. **B** **Fail**: last-50 0, never revises. Details: [`tm051_results.md`](tm051_results.md).

## TM.0.5.2 (unnamed motor / shared return)

Unread W has no motor name. Stamp the act the body just did onto a rare committed note. **A** **Store-works**: after ρ reset W gone → PRESS from `w7=press`; held-out C TUNE from stamped `tune`. **B** **Fail**: shared return last-50 0. Not English NLP. Details: [`tm052_results.md`](tm052_results.md).

## TM.0.5.3 (use-the-fact / shared return)

The file is a fact about this station, not a global motor. **A** **Store-works**: after A life, same S → A PRESS / C HOLD; after C life → C TUNE / A HOLD. Copy-only still PRESS on C. **B** **Fail**: last-50 0. Details: [`tm053_results.md`](tm053_results.md).

## TM.0.5.4 (Open W / shared return)

Unread W is distinct multi-paragraph documents, not cloned one-liners. Same find/stamp/here-match. **A** **Store-works**: after A life, same S → A PRESS / C HOLD; after C life → C TUNE / A HOLD. Copy-only still PRESS on C. **B** **Fail**: last-50 0. Details: [`tm054_results.md`](tm054_results.md).

## TM.0.5.5 (accumulate S / shared return)

Two lives, one store. **A** **Store-works**: after A then C on the same S → A PRESS / C TUNE. Wipe-between loses A. **B** **Fail**: last-50 0. Details: [`tm055_results.md`](tm055_results.md).

## TM.0.5.6 (never-wipe train / shared return)

Training keeps S. **A** **Store-works**: after 500 A lives the dirty store still PRESS; a C life on that S adds TUNE without losing A. **B** **Fail**: last-50 0. Details: [`tm056_results.md`](tm056_results.md).

## TM.0.5.7 (find without unique rare / shared return)

Several distinctive clutter pages, not one unique rare needle. **A** **Store-works**: never-wipe dirty S still PRESS; C life adds TUNE without losing A. **B** **Fail**: C stays HOLD (last-50 0.86 on the first fact). Details: [`tm057_results.md`](tm057_results.md).

## TM.0.5.8 (scale of Open W / shared return)

64 distinct document-shaped clutter pages. **A** **Store-works**: never-wipe dirty S still PRESS; C life adds TUNE without losing A. **B** **Store-works** on this slice (last-50 0.94); not the jump, not retuned. Details: [`tm058_results.md`](tm058_results.md).

## TM.0.5.9 (correct dirty S / shared return)

Stop stamp-collecting on a never-wipe life. **A** **Store-works**: train S is one stamped note and still PRESS; C life adds TUNE without losing A. **B** **Fail**: C stays HOLD (last-50 0.98 on the first fact). Details: [`tm059_results.md`](tm059_results.md).

## TM.0.6.0 (English life / shared return)

Tiny English corpus. Pages never say `press`/`tune`. **A** **Store-works**: after a free life A PRESS / C HOLD from `push` bound in S (`did=press` is bookkeeping); bind-off HOLD. **B** **Store-works** on this slice (last-50 0.46); not the jump, not retuned. Details: [`tm060_results.md`](tm060_results.md).

## TM.0.6.1 (one bind / shared return)

Two rare words on the useful page. **A** **Store-works**: bind=`push` not `argon`; nonce-only S HOLD; bind-all on argon PRESS. **B** **Store-works** on this slice (last-50 0.90); not the jump, not retuned. Details: [`tm061_results.md`](tm061_results.md).

## TM.0.6.2 (never-wipe English / shared return)

Never-wipe train on the one-bind English recipe. **A** **Fail**: dirty train S still PRESS from `push`, but a C life on that S did not stamp TUNE. **B** **Store-works** on this slice (last-50 0.92); not the jump, not retuned. Details: [`tm062_results.md`](tm062_results.md).

## TM.0.6.3 (new-here stamp / shared return)

A growing English store takes a second station as a new unmarked page. **A** **Store-works**: never-wipe train S still PRESS from `push`; C life on that S stamps `bind=adjust` and TUNE. **B** **Store-works** on this slice (last-50 0.92); not the jump, not retuned. Details: [`tm063_results.md`](tm063_results.md).

## TM.0.6.4 (English find without unique rare / shared return)

Same recipe on a multi-rare English W. **A** **Fail**: motors still PRESS/TUNE, but search bound clutter hapax (`xenon`/`neon`/`krypton`) as acts, not only `push`. **B** **Store-works** on the motor bar (last-50 0.94); not the jump, not retuned. Details: [`tm064_results.md`](tm064_results.md).

## TM.0.6.5 (concurrent bind / shared return)

Stamp the attended page, then block extra hapax at this station. **A** **Store-works**: only `bind=push` at A; C life `bind=adjust`. **B** **Fail** (first CS was `neon`; C missed TUNE). Details: [`tm065_results.md`](tm065_results.md).

## TM.0.6.6 (correct dirty English S / shared return)

Here-only + sweep on the English concurrent-bind store. **A** **Fail**: train S is one file `bind=push` (the junk drawer is gone), but C life bound `xenon` not `adjust`. **B** **Store-works** on the motor bar (n=1, last-50 1.00); not the jump, not retuned. Details: [`tm066_results.md`](tm066_results.md).

## TM.0.6.7 (in-hand new-here / shared return)

A new station stamps the attended rare page, not the first leftover rare in W. **A** **Fail**: leftover walk is gone, but C search held `c09` / `neon`, not `p98` / `adjust`. **B** **Store-works** on the motor bar (n=1, last-50 1.00); not the jump, not retuned. Details: [`tm067_results.md`](tm067_results.md).

## TM.0.6.8 (find-novel / shared return)

Search keeps unread pages with the most rare tokens S lacks, and attends that page without stamp-collecting leftover hapax. **A** **Store-works**: train S `bind=push`; C life `bind=adjust`. **B** **Store-works** on this slice (n=1, last-50 0.94); not the jump, not retuned. Details: [`tm068_results.md`](tm068_results.md).

## TM.0.6.9 (find-novel without unique two-rare / shared return)

Same find-novel recipe, but several clutter pages also match the novel-count of `p99`/`p98`. **A** **Fail**: train S `bind=neon` on `c09`; C life `bind=xenon` on `c08`. **B** **Store-works** on the motor bar (n=1, last-50 0.92); not the jump, not retuned. Details: [`tm069_results.md`](tm069_results.md).

