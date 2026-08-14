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

