# Three-memory experiment protocol (v0)

## Question

> Can frozen innate drives + learning rules fill an **inspectable** world-knowledge store from experience, such that facts **survive reset of the working trace** — while the trace alone does not?

## Developmental rule (locked, TM.0.5+)

Jumps must get closer to a machine that **could in theory** learn from wiki / English / an open world. Another isomorphic toy (new room, same digit-copy) is not a jump.

**How** is genome. **What** is a life. Not all at once. Gradual, like a brain.

| Layer | In the recipe (frozen cortex + boxed heads + S/W/ρ) | In experience (later lives) |
|-------|------------------------------------------------------|-----------------------------|
| How to learn | Find unread data, commit, use after ρ reset — without planted answer integers, a door table, or an English lexicon | — |
| How to correct | Detect mismatch, revise S (mark/overwrite/drop), retry, keep the correction after ρ reset | — |
| Learn English | Genome may know **streams of symbols exist** and that the body has named acts. It may not know this corpus is English or that a synonym means a motor | A later life over English pages, tiny corpus first (TM.0.6.0); one bind against a distractor hapax (TM.0.6.1); never-wipe English (TM.0.6.2); new-here stamp (TM.0.6.3); English find without a unique rare token (TM.0.6.4); concurrent bind / block extra hapax here (TM.0.6.5); correct the dirty English store (TM.0.6.6); in-hand new-here (TM.0.6.7) |
| Learn math | Not a calculator in cortex | A later life after some language is already in S |

Four recipe skills, in order: **find / commit / use / correct**. English and math are lives that use those skills. Dumping wiki + algebra into one experiment is illegal.

**Test for every jump:** did we improve the recipe, or smuggle a subject into DNA? A PRESS/TUNE word list in the agent is smuggling English. `+` in the cortex is smuggling math. Split credit is an experimenter crutch; shared return stays an honest arm until the recipe can take one life signal.

**Self-correction (recipe, not a subject):** (1) detect — S and the world disagree; (2) revise — do not only append; (3) retry — look again; (4) keep — after ρ reset the corrected file steers, the first wrong commit does not. Without this, learning is stamp-collecting.

Order of recipe jumps (English is a life that uses the skills, not a lexicon in DNA):

1. No answer integers in W (TM.0.5.0).
2. Use the committed file as text (not int→motor). TM.0.5.0: copy an **innate motor name** mentioned in the page. TM.0.5.2: the page does **not** name the motor; stamp the act the body just did onto a rare committed note. Closed body vocabulary, not English NLP.
3. Search without `has_code` (side-effect of 1 if pages have no place ints; keep it load-bearing). TM.0.5.7: find without a unique rare token (`has_code` stays in the vector). TM.0.5.8: scale of Open W (64 documents). TM.0.6.4: same uniqueness test on the English recipe (hapax clutter; still `{has_code, has_rare}`).
4. Probe is use-the-fact, not only pick-a-motor (TM.0.5.3).
5. Open W (wiki-shaped content) (TM.0.5.4).
6. One return (shared credit that actually works, or an honest genome admission).
7. No `domain=` switch.
8. Accumulate S (stop wiping every episode). TM.0.5.5 eval path; TM.0.5.6 never-wipe train.
9. Correct: wrong commit, world says no, revise S, ρ reset, corrected file works (TM.0.5.1). TM.0.5.9: correct the dirty never-wipe store (stop appending once S names here; drop unstamped pages after a real stamp). TM.0.6.6: same correct flags on the English concurrent-bind store.
10. English life, tiny corpus (TM.0.6.0). One bind per note against a distractor hapax (TM.0.6.1). Never-wipe English on that recipe (TM.0.6.2). New-here stamp so a second station gets an unmarked page (TM.0.6.3). English find without a unique rare token (TM.0.6.4). Concurrent bind: stamp the page in play, block extra hapax at this station (TM.0.6.5). Correct the dirty English store (TM.0.6.6). In-hand new-here: a new station stamps the attended rare page, not the first leftover rare in W (TM.0.6.7). Then more language in S, then a math life.

Do not restore digit-copy, filed `action=`/`where=`, the door toy, or a synonym lexicon to rescue a plot.

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

## v10 free life (no forced curriculum)

Same write box and `{here, that act}` template as v9. W clutter only. During the life, `act(explore=True)` is ε-greedy over percept affordances (no USE_KEY without a held key). Probe is greedy (`explore=False`). **No** OPEN→PICK_KEY→USE_KEY fallback.

Held-out: free green life (must find WAIT without a WAIT tuple).

| ID | If |
|----|----|
| Confound | Cortex moves; disable-S still `use_key`; answer in W; `n_forced > 0`; probe explored |
| Fail | Free red never opens; no file; greedy probe fails; free green never opens |
| Store-works | `n_forced = 0`; S authored from a real opening; red `use_key` and green `wait` after ρ reset; empty S / disable-S fail |

Do not sneak a hidden script. Do not let the policy emit `use_key` / `wait`.

## v11 select among authored notes

One S. Two free lives (red, then green). `n_forced = 0`. W clutter only. Files must be authored, not placed.

Retrieve **select** = tag match on `door=`. Control **dump-all** applies every `action=` bias.

| ID | If |
|----|----|
| Confound | Cortex moves; disable-S still correct; answer in W; `n_forced > 0`; probe explores |
| Fail | Only one file; select red or green wrong; dump-all correct on **both** probes |
| Store-works | Both files authored; select red `use_key` and green `wait`; dump-all fails at least one probe; empty S / disable-S fail |

Do not hand-write the two tags into S. Do not raise store bias so dump-all also works.

## v12 boxed select vs dump

Retrieve is a policy head, not a constructor flag. Features `{n_store ≥ 2, n_hits ≥ 1}` — no door id. Untrained dumps. Train on red+green S (reward on red probe). Held-out: free blue life authors `d1.tag`. Probe greedy.

| ID | If |
|----|----|
| Confound | Cortex moves; disable-S still `use_key`; answer in W; `n_forced > 0`; probe explores |
| Fail | Head unchanged; untrained already correct on red; trained dumps; blue unused; dump-all correct on all probes |
| Store-works | Head changed; untrained dump fails red; trained select red `use_key` / green `wait` / blue `open`; dump-all still fails red |

Do not put door id in retrieve features. Do not let the policy emit the motor act.

## v13 generic copy + use gate

v7–v12 apply a frozen USE_KEY/WAIT table. v13 (`use_read=True`): untrained gate off (ignore the tag). When on: `logits[int(action)] += 3.0` only. Features `{s_hit}` — no door id. Train write+use on red free lives. Held-out green/blue. Retrieve frozen select. W has no answer files. Probe greedy. `n_forced=0`.

| ID | If |
|----|----|
| Confound | Cortex moves; disable-S still `use_key`; answer in W; `n_forced > 0`; probe explores; door id in use features |
| Fail | Use-head unchanged; planted tag already `use_key`; red works, green/blue don’t; empty S still `use_key` |
| Store-works | Cortex frozen; use-head changed; untrained (empty or planted) `open`; trained red `use_key`; held-out green `wait` / blue `open`; empty S / disable-S `open`; dump-all still mixes |

Do not restore the if/elif table on this path. Do not put door id in the use-head. Do not emit `use_key` with no file.

## v14 A pick-one vs B schema

Same cortex. Generic copy. Two arms.

**A.** Unique files (no overwrite). Pick head: one vs all, features `{n_hits ≥ 2}` — no door, no `action=`. When one: frozen newest `when=`. Train on red stale+new. Held-out green conflict.

**B.** Schema head: `{door}` vs `{door, action}`. Integer from the event. Train on red. Held-out green life. Door-only plant must stay `open`.

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; disable-S `use_key`; door in pick features | Cortex moves; disable-S `use_key`; door in schema features |
| Fail | Untrained already `use_key`; green fails; apply-all still works | Untrained already has `action=`; green fails; door-only plant `use_key` |
| Store-works | Untrained mix; trained newest red `use_key`; green `wait` | Untrained door-only `open`; trained complete red `use_key`; green `wait` |

Do not put the motor act in either head. Do not restore the if/elif table.

## v15 joint, no clamps

Unique files + write WHEN + schema + use-gate + pick-one, all learned. No `force_use` / `force_write`. Train on red with a stale wrong note in S. Held-out green with stale `open` in S. Generic copy. Probe greedy. `n_forced=0`. W has no answers.

| ID | If |
|----|----|
| Confound | Cortex moves; disable-S `use_key`; answer in W; probe explores; door id in a head |
| Fail | Untrained already `use_key`; green fails; apply-all still works; a gate was clamped to rescue the plot |
| Store-works | All four heads changed; untrained not `use_key`; trained red `use_key` (newest complete); green `wait`; empty S / disable-S `open`; apply-all mixes |

Do not re-clamp. Do not put the motor act in a head.

## v16 A ok= vs newest, B shared return

**A.** Rank features `{is_newest, has_ok}`. Untrained recency prior. After a success write (`ok=1`), plant newer junk. Train red. Held-out green junk `open`. Newest-prior control must still fail.

**B.** v15 joint setup with **one** return (`r = probe correct`) for every head. No split credit.

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; disable-S `use_key`; door in rank features | Cortex moves; disable-S `use_key` |
| Fail | Untrained already `use_key`; green fails; newest-wins still works | Red still `open`; last-50 ≈ 0 |
| Store-works | Untrained newest wrong; trained `ok=1` red `use_key`; green `wait` | Same as v15 under shared return |

Do not restore newest-wins or split credit to rescue an arm.

## v17 A read do=, B match here=

**A.** Write `do=` not `action=`. Key head: `{action, do}`, features `{s_hit}`. Untrained reads `action=`. Train red. Held-out green `do=0`. `action=` control must fail.

**B.** Write `here=` not `door=`. Match head: `{door, here}`. Untrained matches `door=`. Train red. Held-out green `here=2`. `door=` control must fail.

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; disable-S `use_key`; integer in key head | Cortex moves; disable-S `use_key`; door id in match head |
| Fail | Planted `do=` already `use_key`; green fails; `action=` control works | Planted `here=` already `use_key`; green fails; `door=` control works |
| Store-works | Untrained `open`; red `use_key` from `do=`; green `wait` | Untrained `open`; red `use_key` from `here=`; green `wait` |

Do not restore `action=` / `door=` to rescue a plot.

## v18 A write do=, B write here=

**A.** Read frozen to `do=`. Write-key head: `{action, do}`. Untrained writes `action=`. Train red. Held-out green `do=0`. `action=` control must fail.

**B.** Match frozen to `here=`. Write-place head: `{door, here}`. Untrained writes `door=`. Train red. Held-out green `here=2`. `door=` control must fail.

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; disable-S `use_key`; integer in write-key head | Cortex moves; disable-S `use_key`; door id in write-place head |
| Fail | Untrained already writes `do=`; green fails; freeze-write `do=` | Untrained already writes `here=`; green fails; freeze-write `here=` |
| Store-works | Untrained `open`; red `use_key` from authored `do=`; green `wait` | Untrained `open`; red `use_key` from authored `here=`; green `wait` |

Do not freeze-write `do=` / `here=` or restore `action=` / `door=` read to rescue a plot.

## v19 A shared value-name, B shared place-name

Neither side frozen. Untrained priors disagree: write `action=` / `door=`, read/match `do=` / `here=`. Both heads may move. Either agreed name is legal.

**A.** Write-key + read-key. **B.** Write-place + match-key.

Train red. Held-out green must use the **same** name. Write+use without agreement must stay `open`. Empty-S green must stay `open`.

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; disable-S `use_key`; integer in a name head; empty-S green `wait` | Cortex moves; disable-S `use_key`; door id in a place head; empty-S green `wait` |
| Fail | Untrained already `use_key`; green fails or uses a different name; freeze a head | same |
| Store-works | Untrained `open`; red `use_key` from one shared value name; green `wait` on that name | Untrained `open`; red `use_key` from one shared place name; green `wait` on that name |

Do not freeze one side or restore a wired name to rescue a plot.

## v20 A find unread W, B find vs junk

v9–v19 authored S. v20 puts **find-and-commit** back on the native-tag stack. Collect is the frozen v5 rule (S miss + W hit → commit). Use-gate may learn. Matcher `{door, here}`. Generic copy. No `d0.tag`. No writes from life. Split credit (found `here=` in S / unmount probe). Probe greedy.

**A.** W has `{here:0, action:2}` as `p99.tag` among clutter. Untrained `door=` misses.

**B.** Plus junk `{door:0, action:0}`. Untrained may commit junk; use-gate off stays `open`. Trained `door=` would copy `wait`.

Held-out green: unread `{here:2, action:0}` as `p98.tag`. B also has junk `{door:2, action:1}`.

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; disable-S `use_key`; `d0`/`d2` in W; empty-S green `wait`; writes from life | same |
| Fail | Untrained already `use_key` or `here=`; green fails; freeze-match `here=`; use-on + `door=` still `use_key` | same, plus junk-only `use_key` |
| Store-works | Untrained `open`; unmount W red `use_key` from `p99.tag`; green `wait`; `door=` control `open` | Untrained `open`; red `use_key` from `here=`; green `wait`; `door=` control `wait`; junk-only `open` |

Do not freeze-match `here=` or restore `d0.tag` / the USE_KEY table to rescue a plot.

## v21 A first-file among W hits, B dump-all W hits

v20 found by query name. Collect still kept `w_hits[0]`. v21 freezes `here=` and commit-on-hit. Copy frozen on. Boxed head: filename-first / dump-all vs **newest `when=`**. Features `{n_hits ≥ 2, n_hits ≥ 1}` — no door id, no `action=`. No `d0.tag`. Probe greedy.

**A.** Untrained `aaa.tag` `{here:0, action:0, when:1}` → `wait`. Useful `p99.tag` `when:9`.

**B.** Untrained commits every `here=` hit. Mix → red `wait`.

Held-out green: junk `open` older, `wait` newer. Recency-swap: newest is junk; must not `use_key`.

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; disable-S `use_key`; `d0`/`d2` in W; empty-S green `wait` | same |
| Fail | Untrained already `use_key` or newest; green fails; recency-swap `use_key` | dump already `use_key`; both files remain; recency-swap `use_key` |
| Store-works | Untrained `wait`; unmount red `use_key` from newest; green `wait`; first-file and swap `wait` | Untrained dump `wait`; trained `p99.tag` only; green `wait`; dump and swap `wait` |

Do not freeze-newest or restore filename-first / dump-all to rescue a plot.

## v22 A complete vs stub, B joint no clamps

v21 used planted `when=`. v22 A removes it. v22 B removes v20/v21 clamps.

**A.** No `when=`. Stub `{here:0}` sorts first. Useful `{here:0, action:2}`. Query `here=` and copy frozen. Head: complete vs first. Features `{has_payload, n_hits≥2}`.

**B.** Match + wsel + use together. No `force_use`. No frozen `here=`. W has first/newest `here=` pages and `door=` junk. Split credit.

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; disable-S `use_key`; planted `when=`; `d0` in W | Cortex moves; disable-S `use_key`; `d0` in W |
| Fail | Untrained already `use_key`; complete-is-junk `use_key`; stub-only `use_key` | Untrained already `use_key`; freeze-match / first / use-off still `use_key` |
| Store-works | Untrained stub `open`; red complete `use_key`; green `wait` | Untrained `open`; red newest `here=` `use_key`; green `wait`; three controls fail |

Do not plant `when=` on A or freeze `here=` / `force_use` on B to rescue a plot.

## v23 A joint find+complete+use, B shared return

v22 A had complete-vs-stub without `when=`. v22 B composed find+newest+use under split credit (still used planted recency). v23 puts find + complete + use on the same unread W, no `when=`, and asks if split credit is load-bearing.

**A.** Split: found `here=` / kept `action=2` / unmount probe. W: stub `{here:0}`, complete `{here:0, action:2}`, `door=` junk. Match learns `here=`. No `force_use`.

**B.** Same `make()`. One `r` = unmount probe correct, applied to match, wcomp, and use.

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; disable-S `use_key`; planted `when=`; `d0` in W; empty-S green `wait` | same |
| Fail | Untrained already `use_key`; green fails; freeze-match / stub / use-off still `use_key` | Untrained already `use_key`; red stays `open`; last-50 ≈ 0 |
| Store-works | Untrained `open`; red complete `here=` `use_key`; green `wait`; three freeze-offs fail | Same joint as A without splitting the return |

Do not plant `when=` or restore split credit to rescue B.

## TM.0.1.0 A open query names, B shared return

First post-toy series. Drop the `{door, here}` match menu. Query names are keys that exist in W/S files. Features `{has_hit, key_common}` — no name id. Copy still `action=`. Cortex frozen. No `d0.tag`. No `here=`. No `when=`. Probe greedy.

**A.** Split: found `action=2` in S / unmount probe. Useful unread page `{loc:0, action:2}`. Clutter is `{place, action}`.

**B.** Same `make()`. One `r` = unmount probe correct, applied to qname and use.

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; disable-S `use_key`; planted `when=` / `here=`; `d0` in W; empty-S green `wait` | same |
| Fail | Untrained already `use_key`; green fails; `{door, here}` menu still solves; freeze-qname / use-off still `use_key` | Untrained already `use_key`; red stays `open`; last-50 ≈ 0 |
| Store-works | Untrained `open`; red `use_key` from `loc=` page; green `wait`; menu / qname-off / use-off fail | Same without splitting the return |

Do not restore `{door, here}` or plant `here=` to rescue a plot.

## TM.0.1.1 A open copy names, B shared return

Query frozen to the files’ place key. Drop the `{action, do}` copy menu. Copy names are keys on the hit. Features `{is_query, key_common}` — no name id, no integer. Cortex frozen. No `d0.tag`. No `do=`. No `when=`. Probe greedy.

**A.** Split: chosen copy-key’s value is 2 / unmount probe. Useful unread page `{loc:0, act:2}`. Green `{loc:2, act:0}` — copying the place code would `use_key`.

**B.** Same `make()`. One `r` = unmount probe correct, applied to vname and use.

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; disable-S `use_key`; planted `when=` / `do=`; `d0` in W; empty-S green `wait` | same |
| Fail | Untrained already `use_key`; green fails; `{action, do}` menu still solves; freeze-vname / use-off still `use_key` | Untrained already `use_key`; red stays `open`; last-50 ≈ 0 |
| Store-works | Untrained `open`; red `use_key` from `act=`; green `wait`; menu / vname-off / use-off fail | Same without splitting the return |

Do not restore `{action, do}` or plant `do=` to rescue a plot.

## TM.0.1.2 A messy retrieve, B shared return

No exact `loc=` / `door=` / `here=` query. Rank unread files. Features `{has_code, has_rare}`. Useful page `{where:0, action:2, pad:7}`. Copy frozen `action=`. Cortex frozen. No `d0.tag`. No `when=`. Probe greedy.

**A.** Split: found `action=2` in S / unmount probe.

**B.** Same `make()`. One `r` = unmount probe correct, applied to search and use.

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; disable-S `use_key`; planted `when=` / `loc=` / `here=`; `d0` in W; empty-S green `wait` | same |
| Fail | Untrained already `use_key`; green fails; exact `{door, here}` still solves; freeze-search / use-off still `use_key` | Untrained already `use_key`; red stays `open`; last-50 ≈ 0 |
| Store-works | Untrained `open`; red `use_key` from messy `p99.tag`; green `wait`; exact-match / search-off / use-off fail | Same without splitting the return |

Do not restore exact match or plant `when=` to rescue a plot.

## TM.0.2.0 A scale of W, B shared return

Same messy search as TM.0.1.2. W has **256** unread `.tag` files. Features `{has_code, has_rare}` unchanged. Useful page `{where:0, action:2, pad:7}`. Copy frozen `action=`. Cortex frozen. No `d0.tag`. No `when=`. Probe greedy. `n_train=10000` (predeclared).

**A.** Split: found `action=2` in S / unmount probe.

**B.** Same `make()`. One `r` = unmount probe correct, applied to search and use.

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; disable-S `use_key`; planted `when=` / `loc=` / `here=`; `d0` in W; `w_n < 200`; empty-S green `wait` | same |
| Fail | Untrained already `use_key`; green fails; exact `{door, here}` still solves; freeze-search / use-off still `use_key` | Untrained already `use_key`; red stays `open`; last-50 ≈ 0 |
| Store-works | Untrained `open`; red `use_key` from `p99.tag` in 256-file W; green `wait`; exact-match / search-off / use-off fail | Same without splitting the return |

Do not restore exact match, plant `when=`, shrink W, or retune heads to rescue a plot.

## TM.0.3.0 A free life, B shared return

Not the scripted probe→unmount→probe train. Free life: wander, search W, commit, act. Then ρ reset, W gone, greedy probe. Messy page `{where:0, action:2, pad:7}`. Copy frozen `action=`. Cortex frozen. `write_from_events=False`. `n_forced=0`. `record_search_on_explore` so search leaves traces during life.

**A.** Split: found messy page in S during life / probe after reset.

**B.** Same `make()`. One `r` = probe correct, applied to search and use.

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; disable-S `use_key`; planted `when=` / `loc=` / `here=`; `d0` in W; writes from life; forced curriculum; empty-S green `wait` | same |
| Fail | Untrained probe already `use_key`; green fails; exact match still solves; freeze-search / use-off still `use_key`; training not a free life | Untrained already `use_key`; red stays `open`; last-50 ≈ 0 |
| Store-works | Untrained probe `open`; free red commits `p99.tag`; after reset W gone → `use_key`; green `wait`; controls fail | Same without splitting the return |

Do not restore the scripted unmount train or exact match to rescue a plot.

## TM.0.3.1 A documents free life, B shared return

Same free life as TM.0.3.0. W is `.md` documents with prose + embedded `k=v` (not tidy `.tag` W). Useful page `p99.md` `{where:0, action:2, pad:7}`. Copy frozen `action=`. Cortex frozen. No NOTE-copy. `write_from_events=False`. `n_forced=0`.

**A.** Split: found messy page in S during life / probe after reset.

**B.** Same `make()`. One `r` = probe correct, applied to search and use.

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; disable-S `use_key`; `.tag` in W; no prose; planted `when=` / `loc=` / `here=`; `d0` in W; writes from life; forced curriculum | same |
| Fail | Untrained probe already `use_key`; green fails; exact match still solves; freeze-search / use-off still `use_key`; training not a free life | Untrained already `use_key`; red stays `open`; last-50 ≈ 0 |
| Store-works | Untrained probe `open`; free red commits from `p99.md`; after reset W gone → `use_key`; green `wait`; controls fail | Same without splitting the return |

Do not restore `.tag` W, NOTE-copy, or exact match to rescue a plot.

## TM.0.3.2 A prose retrieve free life, B shared return

Same free life. W is pure prose `.md` with **no** filed `where=` / `action=` / `loc=` / `door=` / `here=`. Digits → anonymous `n*`. Vname picks which int to copy (`is_code` vs not). Cortex frozen. No NOTE-copy. `n_forced=0`.

**A.** Split: found door+motor ints in S during life / probe after reset.

**B.** Same `make()`. One `r` = probe correct.

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; disable-S `use_key`; filed motor/place tags in W; `.tag` in W; `d0` in W; writes from life; forced curriculum | same |
| Fail | Untrained probe already `use_key`; green fails; exact match still solves; freeze search/vname/use still `use_key` | Untrained already `use_key`; red stays `open`; last-50 ≈ 0 |
| Store-works | Untrained probe `open`; free red commits prose ints; after reset W gone → `use_key`; green `wait`; S has `n*` not `action=` | Same without splitting the return |

Do not restore filed `action=` / `where=` or NOTE-copy to rescue a plot.

## TM.0.4.0 A channel dial free life, B shared return

Leave the key/door room. Same prose free life (`n*`, search, vname). World is `ChannelDialWorld` (5 motors). Held-out channel C: place code equals PRESS; must copy TUNE. Species prior HOLD. Cortex frozen (new hash vs door; seed 1337). `n_forced=0`.

**A.** Split: found place+motor ints in S during life / probe after reset.

**B.** Same `make()`. One `r` = probe correct.

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; door world restored; filed motor/place tags; `.tag` W; writes from life; forced curriculum; empty S TUNE on C | same |
| Fail | Untrained already PRESS on A; A life misses pair; after reset not PRESS; C not TUNE; controls still PRESS; swap IDLE still PRESS | Untrained PRESS; A/C miss; empty S solves A; split restored |
| Store-works | Untrained HOLD; free A commits prose ints; after reset W gone → PRESS; C TUNE; S has `n*` | Same without splitting the return |

Do not restore the door toy or filed `action=` to rescue a plot.

## TM.0.5.0 A no answer integers, B shared return

Recipe jump 1: unread W has **no** place/motor digits. Useful page mentions an **innate motor name** (`press` / `tune`), not `0`/`1`/`3`. Search cannot use `has_code`. Vname picks a token; untrained prefers a common word; trained copies the act-name token. Generic copy is name→same-named motor (closed body vocabulary), not an English synonym table, not `logits[int] += 3.0`. Channel dial. Species prior HOLD. Cortex frozen. `n_forced=0`. Writes from life off.

Honest limit: the genome knows the names of its own acts. It does not parse English. Synonyms (`push` for PRESS) are a later English life.

**A.** Split: found act-name token in S during life / probe after reset.

**B.** Same `make()`. One `r` = probe correct.

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; door world; body digits in W; filed tags; `.tag` W; digit-copy (`use_prose_ints`); synonym lexicon; writes from life; forced curriculum; empty S TUNE on C | same |
| Fail | Untrained already PRESS; A life misses `press`; after reset not PRESS; C not TUNE; controls still PRESS; swap `idle` still PRESS | Untrained PRESS; A/C miss; empty S solves A; split restored |
| Store-works | Untrained HOLD; free A commits `press`; after reset W gone → PRESS; C TUNE from `tune`; S has `w*` tokens not `n*` motor ints | Same without splitting the return |

Do not restore answer integers, filed `action=`, or a PRESS/TUNE lexicon beyond innate act names to rescue a plot.

## TM.0.5.1 A correct a wrong commit, B shared return

Recipe: detect fail, drop junk S, blacklist that W page, retry, keep after ρ reset. Same no-integer dial prose as TM.0.5.0. Search **frozen untrained** (first remaining file) so correction is load-bearing — not a rerun of rare-word search. Genome will not drop a file that already names an innate act. Species prior HOLD. Cortex frozen. `n_forced=0`. Writes from life off. Digit-copy off.

**A.** Split: found `press` in S after a life that revised / probe after reset.

**B.** Same `make()`. One `r` = probe correct.

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; search trained to skip correct; body digits; digit-copy; door world; filed tags; writes from life; forced curriculum; empty S TUNE | same |
| Fail | Untrained PRESS or already revises; after reset not PRESS; C not TUNE; revise-off still PRESS; n_revised=0 | Untrained PRESS; A/C miss; empty S solves A; split restored |
| Store-works | Untrained HOLD, n_revised=0; eval life revises ≥1, drops clutter, commits `press`; after ρ reset W gone → PRESS; C TUNE; revise-off HOLD | Same without splitting the return |

Do not restore trained search, answer integers, or “always delete on fail including motor-name notes” to rescue a plot.

## TM.0.5.2 A unnamed motor, B shared return

Recipe jump 2, second slice: unread W has **no** innate motor name and **no** digits. Useful page is a rare-word scrap. On a real success, stamp the body’s act name onto a rare committed note (extra `w*` token). Copy that name after ρ reset. Channel dial. Species prior HOLD. Cortex frozen. `n_forced=0`. v9 writes from life off. Digit-copy off. Revise off (isolate this jump). Search trained (unlike TM.0.5.1).

**A.** Split: find rare page / stamp the act / probe after reset.

**B.** Same `make()`. One `r` = probe correct.

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; motor name still in W; door world; body digits; filed tags; `.tag` W; digit-copy; synonym lexicon; v9 `write_from_events`; forced curriculum; empty S TUNE on C | same |
| Fail | Untrained already PRESS or already stamps; A life misses stamp; after reset not PRESS; C not TUNE; annotate-off / clutter-only still PRESS | Untrained PRESS; A/C miss; empty S solves A; split restored |
| Store-works | Untrained HOLD; free A commits krypton and stamps `press`; after reset W gone → PRESS; C TUNE from stamped `tune` | Same without splitting the return |

Do not restore a motor name in W, digit-copy, filed `action=`, or a PRESS/TUNE lexicon beyond innate act names to rescue a plot.

## TM.0.5.3 A use-the-fact, B shared return

Recipe jump 4: the file is a fact about **this station**, not a global motor. Same unnamed W as TM.0.5.2. On success, stamp act name **and** innate station name (`cha` / `chc`) onto the rare note. After A life, **same S**, ρ reset, W gone: probe A PRESS; probe C HOLD. After C life (fresh S, still wiped between lives): probe C TUNE; probe A HOLD. Copy-only (here-match off) fires PRESS on C — so the match is load-bearing. Channel dial. Species prior HOLD. Cortex frozen. `n_forced=0`. Search still has `has_code`. `domain=` stays. S wiped every episode. Not Open W. Not English.

**A.** Split: find rare page / stamp act+station / probe A correct.

**B.** Same `make()`. One `r` = probe A correct.

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; motor or station name in W; digits; Open W; English; accumulate S; drop `has_code`; drop `domain=`; digit-copy; synonym lexicon; v9 writes; empty S TUNE | same |
| Fail | Untrained PRESS; A's stamp fires on C; C's stamp fires on A; copy-only already HOLDs on C (match not load-bearing) | Untrained PRESS; A miss; split restored |
| Store-works | Untrained HOLD; after A life A PRESS and C HOLD; after C life C TUNE and A HOLD; copy-only PRESS on C | Same without splitting the return |

Do not restore pick-a-motor, a motor name in W, or an English place lexicon to rescue a plot.

## TM.0.5.4 A Open W, B shared return

Recipe jump 5: unread W is **document-shaped** — heading plus a few paragraphs, **distinct** clutter pages (not 11 identical clones). Useful fact still unnamed (no `press`/`tune`/`cha`). Same find/stamp/here-match as TM.0.5.3. After A life, **same S**, ρ reset, W gone: probe A PRESS; probe C HOLD. After C life: probe C TUNE; probe A HOLD. Copy-only still PRESS on C. Clutter words share a closed lexicon so only the useful page is rare (`krypton`/`helium` scrap). Channel dial. Species prior HOLD. Cortex frozen. `n_forced=0`. Search still has `has_code`. `domain=` stays. S wiped every episode. Not English.

**A.** Split: find rare page / stamp act+station / probe A correct.

**B.** Same `make()`. One `r` = probe A correct.

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; cloned one-line W; clutter pages are rare; motor or station name in W; digits; English; accumulate S; drop `has_code`; drop `domain=`; digit-copy; synonym lexicon; v9 writes; empty S TUNE | same |
| Fail | W not Open W; Untrained PRESS; A's stamp fires on C; C's stamp fires on A; copy-only already HOLDs on C | Untrained PRESS; A miss; split restored |
| Store-works | Untrained HOLD; after A life A PRESS and C HOLD; after C life C TUNE and A HOLD; copy-only PRESS on C; distinct multi-paragraph W | Same without splitting the return |

Do not restore cloned one-liners, a motor name in W, or an English lexicon to rescue a plot.

## TM.0.5.5 A accumulate S, B shared return

Recipe jump 8, eval slice: two lives, **same S**, one unread library with both useful pages (krypton and helium) plus distinct Open W clutter. Life A then life C, no `rmtree`. After both, ρ reset, W gone: probe A PRESS (first fact kept) and probe C TUNE (second fact added). Wipe-between control: A HOLD after the second life — accumulate is load-bearing. After A only, C still HOLD (here-match). Copy-only still PRESS on C. Train still wipes each episode. Channel dial. Species prior HOLD. Cortex frozen. `n_forced=0`. Search still has `has_code`. `domain=` stays. Not English.

Genome grammar for two facts (not English): skip already-owned W pages only when S already names a **different** station; do not stamp a note that names another station; retrieve the S file that names **here**.

**A.** Split: find rare page / stamp act+station / probe A correct. Eval: A then C on the same S.

**B.** Same `make()`. One `r` = probe A correct.

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; eval W missing p98; cloned W; motor or station name in W; digits; English; train also never wipes; drop `has_code`; drop `domain=`; digit-copy; synonym lexicon; v9 writes; empty S TUNE | same |
| Fail | S wiped between eval lives; after both lives A not PRESS or C not TUNE; wipe-between still PRESS on A; copy-only already HOLDs on C | Untrained PRESS; two-life miss; split restored |
| Store-works | Untrained HOLD; after A life A PRESS / C HOLD; after both lives A PRESS / C TUNE; wipe-between A HOLD; copy-only PRESS on C | Same without splitting the return |

Do not restore wiping S on the eval path, a motor name in W, or an English lexicon to rescue a plot.

## TM.0.5.6 A never-wipe train, B shared return

Recipe jump 8, train slice: **do not rmtree S each train episode**. ρ still resets (session). After 500 A lives the same store is probed (no fresh A life): PRESS, foil C HOLD. Then a C life on that **dirty** S with both useful pages: A PRESS kept, C TUNE added. Wipe-between loses A. Open W, unnamed pages, here-match. Cortex frozen. `n_forced=0`. Search still has `has_code`. `domain=` stays. Not English. `use_commit_rare_only` on this slice only (do not vacuum clutter once S already names here) — off by default so TM.0.5.5 B stays Fail.

**A.** Split: search rare / write stamp / probe A correct. Train keeps S.

**B.** Same `make()`. One `r` = probe A correct.

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; train still wipes; English; drop `has_code`; drop `domain=`; digit-copy; synonym lexicon; v9 writes; empty S TUNE; rare-commit filter smuggled into 0.5.5 | same |
| Fail | After train, dirty S not PRESS; C life on dirty S loses A or misses TUNE; wipe-between still PRESS on A | Untrained PRESS; dirty S miss; split restored |
| Store-works | Untrained HOLD; after never-wipe train A PRESS / C HOLD; C life on that S A PRESS / C TUNE; wipe-between A HOLD | Same without splitting the return |

Do not restore per-episode train wipe, a motor name in W, or an English lexicon to rescue a plot.

## TM.0.5.7 A find without unique rare, B shared return

Recipe: unread W has **several distinctive clutter pages** (hapax scraps), not one unique rare needle. `{has_code, has_rare}` still ranks; `has_rare` is true for xenon/argon/neon clutter **and** krypton/helium. A distinctive page is a stampable note — uniqueness is not required. Same never-wipe train as TM.0.5.6. Closed-lexicon clutter-only control (no hapax) must not PRESS. Cortex frozen. `n_forced=0`. Search still has `has_code`. `domain=` stays. Not English.

**A.** Split: search rare / write stamp / probe A correct. Train keeps S. W is multi-rare.

**B.** Same `make()`. One `r` = probe A correct.

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; useful page is still the only rare token; English; drop `has_code`; drop `domain=`; digit-copy; synonym lexicon; v9 writes; empty S TUNE | same |
| Fail | After train, dirty S not PRESS; C life on dirty S loses A or misses TUNE; wipe-between still PRESS on A; closed clutter-only PRESS | Untrained PRESS; dirty S miss; split restored |
| Store-works | Untrained HOLD; multi-rare W; after never-wipe train A PRESS / C HOLD; C life on that S A PRESS / C TUNE; wipe-between A HOLD | Same without splitting the return |

Do not restore a unique-rare needle, a motor name in W, or an English lexicon to rescue a plot.

## TM.0.5.8 A scale of Open W, B shared return

Recipe: unread W is a **pile** of document-shaped pages (64 distinct clutter logs), not a dozen. Same multi-rare never-wipe recipe as TM.0.5.7: hapax clutter (`xenon`/`argon`/`neon`) plus krypton/helium, so uniqueness is still gone. Closed-lexicon clutter-only control (64 pages, no hapax) must not PRESS. Cortex frozen. `n_forced=0`. Search still has `has_code`. `domain=` stays. Not English. Do not retune `n_train`.

**A.** Split: search rare / write stamp / probe A correct. Train keeps S. W is scaled.

**B.** Same `make()`. One `r` = probe A correct.

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; W still a dozen logs; useful page is the only rare token; English; drop `has_code`; drop `domain=`; digit-copy; synonym lexicon; v9 writes; empty S TUNE | same |
| Fail | After train, dirty S not PRESS; C life on dirty S loses A or misses TUNE; wipe-between still PRESS on A; closed clutter-only PRESS | Untrained PRESS; dirty S miss; split restored |
| Store-works | Untrained HOLD; 64-page W; after never-wipe train A PRESS / C HOLD; C life on that S A PRESS / C TUNE; wipe-between A HOLD | Same without splitting the return |

Do not shrink W, restore a unique-rare needle, or add an English lexicon to rescue a plot.

## TM.0.5.9 A correct dirty S, B shared return

Recipe: the never-wipe store must not only append. Same 64-page multi-rare W as TM.0.5.8. Once S names **here**, do not commit more W pages for that station. After a successful stamp, drop committed pages that never received an act name. Revise and here-only are **off by default**. Cortex frozen. `n_forced=0`. Search still has `has_code`. `domain=` stays. Not English. Do not retune `n_train`. Predeclared: train S has ≤8 files (TM.0.5.8 had 19).

**A.** Split: search rare / write stamp / revise junk / probe A correct. Train keeps S.

**B.** Same `make()`. One `r` = probe A correct.

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; W still a dozen logs; English; drop `has_code`; drop `domain=`; digit-copy; synonym lexicon; v9 writes; empty S TUNE; revise/here-only smuggled into 0.5.8 | same |
| Fail | After train, S not PRESS; n files > 8; n_revised=0; C life loses A or misses TUNE; wipe-between still PRESS on A | Untrained PRESS; dirty S miss; split restored |
| Store-works | Untrained HOLD; 64-page W; after never-wipe train A PRESS / C HOLD, S small, n_revised≥1; C life on that S A PRESS / C TUNE; wipe-between A HOLD | Same without splitting the return |

Do not restore stamp-collecting, shrink W, or add an English lexicon to rescue a plot.

## TM.0.6.0 A English life, B shared return

Recipe: a later life over English pages, tiny corpus first. Genome may know streams of symbols exist and that the body has named acts. It may not know this corpus is English or that a synonym means a motor. Unread W never says `press`/`tune`. On success, stamp `did=` (skipped by copy) and keep rare page words; aliases in S map those words to the act just done. Alias-bind and did-stamp are **off by default**. Cortex frozen. `n_forced=0`. Search still has `has_code`. `domain=` stays. Not math. Do not retune `n_train`. Bind-off (bookkeeping without lookup) must HOLD.

**A.** Split: search rare / write stamp / copy the bound page word / probe A correct.

**B.** Same `make()`. One `r` = probe A correct.

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; W names an innate motor; nonce scrap; synonym table in the agent; drop `has_code`; drop `domain=`; digit-copy; `w*=press` still the copy token | same |
| Fail | Untrained PRESS; after A life not PRESS; bind-off still PRESS; C miss | Untrained PRESS; A miss; C miss; split restored |
| Store-works | Untrained HOLD; tiny English W; after a free life A PRESS / C HOLD from a page word bound in S; C life TUNE; bind-off HOLD | Same without splitting the return |

Do not restore an innate-name stamp, a `push` table in the agent, or Wikipedia-scale W to rescue a plot.

## TM.0.6.1 A one bind, B shared return

Recipe: a page is many words. Alias **one** rare token per successful note — the first rare word in stream order — not every hapax. Useful pages have a synonym and a distractor that sorts earlier in the alphabet (`argon` before `push`; `alpha` before `adjust`). One-bind is **off by default**. Cortex frozen. `n_forced=0`. Search still has `has_code`. `domain=` stays. Not math. Do not retune `n_train`. Nonce-only S must HOLD; bind-all on that S must PRESS.

**A.** Split: search rare / write one bind / copy the bound word / probe A correct.

**B.** Same `make()`. One `r` = probe A correct.

| ID | A | B |
|----|---|---|
| Confound | Cortex moves; unique-rare needle; synonym table in the agent; drop `has_code`; drop `domain=`; digit-copy; alphabet bind | same |
| Fail | Untrained PRESS; bind=distractor; nonce-only PRESS; bind-all nonce HOLD | Untrained PRESS; A miss; C miss; nonce PRESS |
| Store-works | Untrained HOLD; two rares; bind=synonym; A PRESS / C HOLD; nonce HOLD; bind-all nonce PRESS; C TUNE | Same without splitting the return |

Do not restore bind-all-rares, a unique-rare needle, or a `push` table in the agent to rescue a plot.

## TM.0.6.2 A never-wipe English, B shared return

Recipe: English uses the recipe skills already bought. Same 0.6.1 one-bind W (two rares per useful page). Train **does not rmtree S**. After train, probe that dirty store (no fresh A life). Then a C life on the same S with both pages. `use_commit_rare_only` on this slice only. Revise / here-only stay **off**. Cortex frozen. `n_forced=0`. Search still has `has_code`. `domain=` stays. Not math. Do not retune `n_train`. Nonce-only S must HOLD; bind-all on that S must PRESS.

**A.** Split: search rare / write one bind / copy the bound word / probe A correct. Never wipe train S.

**B.** Same `make()`. One `r` = probe A correct.

| ID | A | B |
|----|---|----|
| Confound | Cortex moves; unique-rare needle; synonym table in the agent; drop `has_code`; drop `domain=`; revise/here-only on; train still wipes | same |
| Fail | Untrained PRESS; train S not PRESS; C life loses A or misses TUNE; wipe-between still PRESS; bind=distractor; nonce PRESS | Untrained PRESS; A miss; C miss; nonce PRESS |
| Store-works | Untrained HOLD; never-wipe train A PRESS from `push`; C life on that S A PRESS / C TUNE; wipe-between A HOLD; nonce HOLD | Same without splitting the return |

Do not restore a train wipe, a unique-rare needle, or a `push` table in the agent to rescue a plot.

## TM.0.6.3 A new-here stamp, B shared return

Recipe: a growing store must take a **new place** as a new unmarked rare page. Same never-wipe one-bind English W as TM.0.6.2. If S already names some other station, a success here stamps an unmarked rare note (commit one if needed) — not a second helping of the A-trained write head. Probe still HOLDs at an unnamed station. New-here is **off by default**. Cortex frozen. `n_forced=0`. Search still has `has_code`. `domain=` stays. Not math. Do not retune `n_train`.

**A.** Split: search rare / write one bind / copy the bound word / probe A correct. Never wipe train S. C life on that dirty S must TUNE.

**B.** Same `make()`. One `r` = probe A correct.

| ID | A | B |
|----|---|----|
| Confound | Cortex moves; unique-rare needle; synonym table in the agent; drop `has_code`; drop `domain=`; probe TUNE without a C bind; revise/here-only on | same |
| Fail | Untrained PRESS; train S not PRESS; C life loses A or misses TUNE; wipe-between still PRESS; nonce PRESS | Untrained PRESS; A miss; C miss; nonce PRESS |
| Store-works | Untrained HOLD; never-wipe train A PRESS from `push`; C life on that S A PRESS / C TUNE from `adjust`; wipe-between A HOLD; nonce HOLD | Same without splitting the return |

Do not restore a train wipe, raise `n_train`, or a `push` table in the agent to rescue a plot.

## TM.0.6.4 A English find without unique rare, B shared return

Recipe: unread English W has **several distinctive clutter pages** (hapax English words), not one unique rare needle at `p99`. Same never-wipe one-bind + new-here recipe as TM.0.6.3. Hapax are `xenon` / `neon` / `krypton` on late clutter files — not `argon` (already on the useful A page). `{has_code, has_rare}` still ranks; uniqueness is gone. Closed-lexicon clutter-only control (no hapax) must not PRESS. Cortex frozen. `n_forced=0`. Search still has `has_code`. `domain=` stays. Not math. Do not retune `n_train`. Do not add a ranker in this slice.

**A.** Split: search rare / write one bind / copy the bound word / probe A correct. Never wipe train S. Bind must be `push`, not a clutter hapax.

**B.** Same `make()`. One `r` = probe A correct.

| ID | A | B |
|----|---|----|
| Confound | Cortex moves; useful page is still the only rare token; synonym table in the agent; drop `has_code`; drop `domain=`; argon as clutter hapax; revise/here-only on | same |
| Fail | Untrained PRESS; train S not PRESS; C life loses A or misses TUNE; wipe-between still PRESS; bind=clutter hapax; nonce PRESS; closed clutter-only PRESS | Untrained PRESS; A miss; C miss; nonce PRESS |
| Store-works | Multi-rare English W; never-wipe train A PRESS from `push` only; C life on that S A PRESS / C TUNE from `adjust`; wipe-between A HOLD; nonce HOLD | Same without splitting the return |

Do not restore a unique-rare needle, a `push` table in the agent, or a new ranker to rescue a plot.

## TM.0.6.5 A concurrent bind, B shared return

Recipe: the CS is the **attended page** when the body succeeds; once this station has a bind, do not stamp a second hapax. Same English multi-rare W as TM.0.6.4. Not a ranker. Not unique-rare restored. Concurrent-bind is **off by default**. Cortex frozen. `n_forced=0`. Search still has `has_code`. `domain=` stays. Not math. Do not retune `n_train`. Do not turn on 0.5.9 here-only/revise in this slice.

**A.** Split: search rare / write one bind / copy the bound word / probe A correct. Never wipe train S. Bind must be `push`, not a clutter hapax.

**B.** Same `make()`. One `r` = probe A correct.

| ID | A | B |
|----|---|----|
| Confound | Cortex moves; unique-rare restored; synonym table in the agent; drop `has_code`; drop `domain=`; argon as clutter hapax; revise/here-only on | same |
| Fail | Untrained PRESS; train S not PRESS; C life loses A or misses TUNE; wipe-between still PRESS; bind=clutter hapax; nonce PRESS | Untrained PRESS; A miss; C miss; nonce PRESS |
| Store-works | Multi-rare English W; one CS here from the page in play; train A PRESS from `push`; C life A PRESS / C TUNE from `adjust`; wipe-between A HOLD; nonce HOLD | Same without splitting the return |

Do not restore unique-rare, a `push` table in the agent, or a ranker to rescue a plot.

## TM.0.6.6 A correct dirty English S, B shared return

Recipe: the never-wipe English store must not only append. Same concurrent-bind multi-rare W as TM.0.6.5. Once S names **here**, do not commit more W pages for that station. After a successful stamp, drop committed pages that never received an act name. New-here stays rare-only (a common in-hand page is not a CS). Revise and here-only are **off by default**. Cortex frozen. `n_forced=0`. Search still has `has_code`. `domain=` stays. Not math. Do not retune `n_train`. Do not restore unique-rare.

**A.** Split: search rare / write one bind / copy the bound word / probe A correct. Never wipe train S. Train S must be small. Bind must be `push`, not a clutter hapax. C life on that S must TUNE from `adjust`.

**B.** Same `make()`. One `r` = probe A correct.

| ID | A | B |
|----|---|----|
| Confound | Cortex moves; unique-rare restored; synonym table in the agent; drop `has_code`; drop `domain=`; argon as clutter hapax; n_train raised | same |
| Fail | Untrained PRESS; train S not PRESS; train S still stamp-collecting; C life loses A or misses TUNE; wipe-between still PRESS; bind=clutter hapax; nonce PRESS | Untrained PRESS; A miss; C miss; n too large; nonce PRESS |
| Store-works | Multi-rare English W; small never-wipe S; train A PRESS from `push`; C life A PRESS / C TUNE from `adjust`; wipe-between A HOLD; nonce HOLD | Same without splitting the return |

Do not restore the junk drawer, a unique-rare needle, or a `push` table in the agent to rescue a plot.

## TM.0.6.7 A in-hand new-here, B shared return

Recipe: a new station stamps the **attended** rare page, or stamps nothing. Same corrected concurrent-bind multi-rare W as TM.0.6.6. Walking unread W in file order is not a CS. In-hand new-here is **off by default**. Cortex frozen. `n_forced=0`. Search still has `has_code`. `domain=` stays. Not math. Do not retune `n_train`. Do not restore unique-rare. Do not rank `p98`.

**A.** Split: search rare / write one bind / copy the bound word / probe A correct. Never wipe train S. Train S must be small. Bind must be `push`, not a clutter hapax. C life on that S must TUNE from `adjust` on the attended page.

**B.** Same `make()`. One `r` = probe A correct.

| ID | A | B |
|----|---|----|
| Confound | Cortex moves; unique-rare restored; synonym table in the agent; drop `has_code`; drop `domain=`; argon as clutter hapax; leftover walk kept; n_train raised | same |
| Fail | Untrained PRESS; train S not PRESS; train S still stamp-collecting; C life loses A or misses TUNE; wipe-between still PRESS; bind=clutter hapax; nonce PRESS | Untrained PRESS; A miss; C miss; n too large; nonce PRESS |
| Store-works | Multi-rare English W; small never-wipe S; train A PRESS from `push`; C life A PRESS / C TUNE from `adjust` on the attended page; wipe-between A HOLD; nonce HOLD | Same without splitting the return |

Do not restore the leftover W walk, a unique-rare needle, or a `push` table in the agent to rescue a plot.


