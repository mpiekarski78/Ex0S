# Comparison to BDH Category B

BDH is the **trace-only baseline**, not a scoreboard.  
Source: [mpiekarski78/bdh](https://github.com/mpiekarski78/bdh) · [docs/conclusion.md](https://github.com/mpiekarski78/bdh/blob/main/docs/conclusion.md)

## v0 (key/door) — same questions, not the same numbers

| Check | BDH (measured) | three-memory S **off** | three-memory S **on** |
|-------|----------------|------------------------|------------------------|
| Slow weights after experience | unchanged | unchanged | unchanged |
| Working trace diverges / moves | yes (ρ L2 ≫ 0) | yes (session success action + ρ) | yes |
| Same probe, different behavior | yes (fragile) | yes before ρ reset (`use_key`) | yes |
| Reset ρ, keep everything else | effect gone | effect gone (`open`) | **effect remains** (`use_key`) |
| Reset S (or never write S) | n/a | n/a | effect gone |
| Snapshot restore of ρ | exact | action match after restore | action match (knowledge does not *need* it) |
| Inspectable record of the fact | no | no | **yes** (`store_A.json`) |

## v1 (language) — same probes, published BDH numbers

Full table: [`v1_results.md`](v1_results.md).

| Check | BDH (published) | three-memory S off | three-memory S on |
|-------|-----------------|--------------------|-------------------|
| Probe | `my lo` → r/v | same | same |
| Empty prior P(v) | 0.181 | 0.027 (stripped pretrain) | 0.027 |
| 8× love, P(v) after ρ reset | wiped | 0.027 (prior) | **0.988** |
| disable-S after ρ reset | n/a (no S) | prior | — |
| Inspectable | no | no | `my lo -> v` |

## Classification

| System | Letter / label |
|--------|----------------|
| Public BDH | **Category B** — short-term adaptive memory |
| three-memory, S off | **Trace-only** (same idea as BDH B) |
| three-memory, S on (v0 and v1 NOTE) | **Store-works** |
| three-memory, S on (v2 raw retrieve) | **Trace-only** |
| three-memory, S = `.md` files (v3 note) | **Store-works** (same as v1; new process) |
| three-memory, S = `.md` files (v3 raw) | **Trace-only** (same as v2; new process) |
| three-memory, v4 note select / v5 note commit | **Store-works** |
| three-memory, v4/v5 raw | **Fail** (file taken, unused) |
| three-memory, v6 tool (plain prior) | **Store-works** (skill in machinery) |
| three-memory, v6 fewshot / untaught NOTE | **Fail** |
| three-memory, v7 native tags | **Store-works** (no English prior) |
| three-memory, v8 boxed policy | **Store-works** (skill box moves; cortex frozen) |
| three-memory, v9 write-from-life | **Store-works** (note authored from events; W has no answer) |
| three-memory, v10 free life | **Store-works** (`n_forced=0`; live then write) |
| three-memory, v11 select authored notes | **Store-works** (pick the matching life; dump-all mixes lives) |
| three-memory, v12 learn select vs dump | **Store-works** (head learns not to dump; held-out blue `open`) |
| three-memory, v13 copy action= | **Store-works** (gate learns to read the integer; held-out green `wait`) |
| three-memory, v14 pick vs schema | **Store-works** / **Store-works** (newest among matches; include `action=` in the note) |
| three-memory, v15 joint no clamps | **Store-works** (write+schema+use+pick together; held-out green `wait`) |
| three-memory, v16 ok= vs newest / shared return | **Store-works** / **Fail** (prefer `ok=1`; one return starves the joint) |
| three-memory, v17 do= / here= | **Store-works** / **Store-works** (learn the field name to copy or match) |
| three-memory, v18 write do= / here= | **Store-works** / **Store-works** (learn the field name to emit) |
| three-memory, v19 shared name | **Store-works** / **Store-works** (write and read learn a convention) |
| three-memory, v20 find in W | **Store-works** / **Store-works** (query `here=`; unread `p99.tag`; junk on `door=` does not leak `use_key`) |
| three-memory, v21 select among W hits | **Store-works** / **Store-works** (newest `when=` over filename-first or dump-all) |
| three-memory, v22 complete vs stub / joint | **Store-works** / **Store-works** (payload over stub; find+pick+use together) |
| three-memory, v23 joint wiki / shared return | **Store-works** / **Fail** (no `when=`; split credit load-bearing on unread W) |
| three-memory, TM.0.1.0 open query names | **Store-works** / **Store-works** (keys from files; `{door, here}` off) |
| three-memory, TM.0.1.1 open copy names | **Store-works** / **Store-works** (keys from the hit; `{action, do}` off) |
| three-memory, TM.0.1.2 messy retrieve | **Store-works** / **Store-works** (rank files; exact `loc=`/`door=` misses) |
| three-memory, TM.0.2.0 scale of W | **Store-works** / **Store-works** (256 unread files; same search) |
| three-memory, TM.0.3.0 a life | **Store-works** / **Fail** (free life find/commit; shared return starves) |
| three-memory, TM.0.3.1 documents | **Store-works** / **Fail** (`.md` W; free life; shared return starves) |
| three-memory, TM.0.3.2 prose retrieve | **Store-works** / **Fail** (no filed `action=`; anonymous `n*`; shared return starves) |
| three-memory, TM.0.4.0 channel dial | **Store-works** / **Fail** (left door world; C TUNE; shared return clutter) |
| three-memory, TM.0.5.0 no answer integers | **Store-works** / **Fail** (no digits in W; innate name token; shared return starves) |
| three-memory, TM.0.5.1 correct | **Store-works** / **Fail** (drop junk S; shared return never revises) |
| three-memory, TM.0.5.2 unnamed motor | **Store-works** / **Fail** (no motor name in W; stamp from the event; shared return starves) |
| three-memory, TM.0.5.3 use-the-fact | **Store-works** / **Fail** (same S: A PRESS, C HOLD; shared return starves) |
| three-memory, TM.0.5.4 Open W | **Store-works** / **Fail** (distinct documents; same S: A PRESS, C HOLD; shared return starves) |
| three-memory, TM.0.5.5 accumulate S | **Store-works** / **Fail** (two lives, same S: A PRESS, C TUNE; shared return starves) |
| three-memory, TM.0.5.6 never-wipe train | **Store-works** / **Fail** (dirty train S still PRESS; C life adds TUNE; shared return starves) |
| three-memory, TM.0.5.7 find without unique rare | **Store-works** / **Fail** (several hapax clutter pages; C life adds TUNE; shared return misses C) |
| three-memory, TM.0.5.8 scale of Open W | **Store-works** / **Store-works** (64-page pile; C life adds TUNE; shared return not the jump) |
| three-memory, TM.0.5.9 correct dirty S | **Store-works** / **Fail** (one stamped note; C life adds TUNE; shared return misses C) |
| three-memory, TM.0.6.0 English life | **Store-works** / **Store-works** (page word bound in S, not a DNA synonym; shared return not the jump) |
| three-memory, TM.0.6.1 one bind | **Store-works** / **Store-works** (distractor hapax on the note does not fire; shared return not the jump) |
| three-memory, TM.0.6.2 never-wipe English | **Fail** / **Store-works** (dirty train S still PRESS from `push`; C life missed TUNE; shared return not the jump) |
| three-memory, TM.0.6.3 new-here stamp | **Store-works** / **Store-works** (second station gets an unmarked page; shared return not the jump) |
| three-memory, TM.0.6.4 English find without unique rare | **Fail** / **Store-works** (clutter hapax bound as acts; motors still work; shared return not the jump) |
| three-memory, TM.0.6.5 concurrent bind | **Store-works** / **Fail** (one CS here from the page in play; shared return first-CS `neon`, C miss) |
| three-memory, TM.0.6.6 correct dirty English S | **Fail** / **Store-works** (train S n=1 `push`; C bound `xenon`; shared return not the jump) |
| three-memory, TM.0.6.7 in-hand new-here | **Fail** / **Store-works** (leftover walk gone; C bound `neon` in-hand; shared return not the jump) |
| three-memory, TM.0.6.8 find-novel | **Store-works** / **Store-works** (C bound `adjust`; shared return not the jump) |
| three-memory, TM.0.6.9 find-novel without unique two-rare | **Fail** / **Store-works** (train bound `neon`, C bound `xenon`; shared return not the jump) |

## Honest limits

- v0 retrieve was tag→action. v1 retrieve is a NOTE string the frozen LM was trained to copy.
- v2 removes that lesson: raw snippet in the prompt. This tiny LSTM does not use it after ρ reset.
- Empty priors differ because lord/love were **stripped** from pretrain on purpose.
- One filler byte does not scramble v1’s prefix-keyed session buffer; BDH’s mixed ρ can. **Reset of ρ** is the matched test.

## v2 (raw retrieve, no NOTE-copy)

| Check | BDH | v1 S on (NOTE) | v2 S on (raw) | v2 S off |
|-------|-----|----------------|---------------|----------|
| Taught retrieve format | n/a | yes | **no** | no |
| P(v) after ρ reset | wiped | 0.988 | **0.093** | 0.084 |
| Classification | B | Store-works | **Trace-only** | Trace-only |

Without a taught use-protocol, this frozen tiny LM + S looks like BDH again. See [`v2_results.md`](v2_results.md).

## v3 (markdown files, no RAG)

S is `.md` on disk. Probe after a **new agent** loads the folder (ρ empty).

| Check | v1 JSON S | v3 note, reload .md | v3 raw, reload .md |
|-------|-----------|---------------------|---------------------|
| Inspectable | JSON | **file** `# my lo` / `my love` | **same file** |
| P(v) after reload | n/a (in-process) | **0.988** | **0.093** |
| JS(reload, in-process) | n/a | **0** | **0** |
| Classification | Store-works | **Store-works** | **Trace-only** |

Disk does not fix v2. It makes S something you can open in an editor. See [`v3_results.md`](v3_results.md).

## v4 / v5 (select and collect)

| Check | v4 note select | v4 note dump-all | v5 note commit, unmount W | v5 note peek, unmount W |
|-------|----------------|------------------|---------------------------|-------------------------|
| P(v) | **0.988** | **0.007** | **0.988** | **0.027** (prior) |
| Class | Store-works | control (hurts) | Store-works | not memory |

S grows → must select. W is available, not known, until commit. Raw arms Fail (same unused-file ceiling as v2). See [`v4_results.md`](v4_results.md), [`v5_results.md`](v5_results.md).

## v6 (use-skill, plain prior)

No NOTE-copy in weights. Fact in `.md`. Skill in machinery.

| Check | tool | fewshot | untaught NOTE |
|-------|------|---------|----------------|
| P(v) after commit, unmount W | **0.649** | 0.053 | 0.053 |
| `love` in LM window | **no** | yes | yes |
| Class | **Store-works** | Fail | Fail |

See [`v6_results.md`](v6_results.md).

## v7 (native tags, no English)

Key/door with integer `.tag` files. Empty prior: `open`. After commit/reload: `use_key`. Dump-all: `open`. Peek-unmount: `open`. **Store-works**. See [`v7_results.md`](v7_results.md).

## v8 (boxed use-policy, frozen cortex)

Policy learns when to commit/apply. Cortex hash unchanged. Action still from `action=` in the file.

| Check | Untrained | After train, unmount W |
|-------|-----------|------------------------|
| Red (`d0.tag`) | `open` | **`use_key`** |
| Held-out green (`d2.tag`) | `open` | **`wait`** |
| Empty S / disable-S | `open` | `open` |
| Class | — | **Store-works** |

See [`v8_results.md`](v8_results.md).

## v9 (write from a life, no answer in W)

Policy learns when to author S. Collect off. W is clutter only.

| Check | Untrained | After train, ρ reset |
|-------|-----------|----------------------|
| Red life → `d0.tag` | S empty, `open` | **authored**, **`use_key`** |
| Held-out green → `d2.tag` | — | **authored**, **`wait`** |
| Empty S / disable-S | `open` | `open` |
| Answer in W | no | no |
| Class | — | **Store-works** |

See [`v9_results.md`](v9_results.md).

## v10 (free life, no script)

ε-greedy over affordances during the life. Probe greedy. No forced curriculum.

| Check | Result |
|-------|--------|
| n_forced | **0** |
| Red life | `pick_key … use_key` (not the v9 tuple), authored `d0.tag`, greedy probe **`use_key`** |
| Green life | sampled `wait`, authored `d2.tag`, greedy probe **`wait`** |
| Empty S / disable-S | `open` |
| Class | **Store-works** |

See [`v10_results.md`](v10_results.md).

## v11 (select among authored notes)

Two free lives. Files authored, not placed. Select vs dump-all.

| Check | Select | Dump-all |
|-------|--------|----------|
| Red | **`use_key`** | **`wait`** |
| Green | **`wait`** | `wait` |
| S | `d0.tag`, `d2.tag` | same pile |
| Class | **Store-works** | control (hurts red) |

See [`v11_results.md`](v11_results.md).

## v12 (learn select vs dump)

Retrieve head. Features are pile-size and match-count, not door id.

| Check | Untrained | Trained | Dump-all |
|-------|-----------|---------|----------|
| Red | dump, `wait` | **select**, **`use_key`** | `wait` |
| Green | — | **select**, **`wait`** | `wait` |
| Held-out blue | — | **select**, **`open`** | `wait` |
| Class | — | **Store-works** | control |

See [`v12_results.md`](v12_results.md).

## v13 (copy `action=` from the file)

Use-gate. Features are match/no-match, not door id. Copy is generic `logits[int(action)] += 3.0`.

| Check | Untrained | Trained | Dump-all |
|-------|-----------|---------|----------|
| Planted `d0.tag` | `open` (gate off) | — | — |
| Red | `open` | **`use_key`** (`action=2`) | `wait` |
| Held-out green | — | **`wait`** (`action=0`) | — |
| Held-out blue | — | **`open`** (`action=1`) | — |
| Class | — | **Store-works** | control |

See [`v13_results.md`](v13_results.md).

## v14 (pick-one vs write schema)

Same cortex. Two heads.

| Check | A pick-one | B schema |
|-------|------------|----------|
| Untrained red | mix `wait` | door-only `open` |
| Trained red | **`use_key`** (newest file) | **`use_key`** (`action=2`) |
| Held-out green | **`wait`** | **`wait`** |
| Class | **Store-works** | **Store-works** |

See [`v14_results.md`](v14_results.md).

## v15 (joint, no clamps)

All four heads. No `force_use` / `force_write`.

| Check | Untrained | Trained | Apply-all |
|-------|-----------|---------|-----------|
| Red | `open` (use-gate off) | **`use_key`** (newest complete) | `wait` |
| Held-out green | — | **`wait`** | — |
| Class | — | **Store-works** | control |

See [`v15_results.md`](v15_results.md).

## v16 (ok= vs newest, shared return)

| Check | A rank `ok=` | B shared return |
|-------|--------------|-----------------|
| Untrained red | newest `wait` | `open` |
| Trained red | **`use_key`** | `open` |
| Held-out green | **`wait`** | `open` |
| Class | **Store-works** | **Fail** |

See [`v16_results.md`](v16_results.md).

## v17 (read `do=` vs match `here=`)

| Check | A `do=` | B `here=` |
|-------|---------|-----------|
| Untrained | `open` | `open` |
| Trained red | **`use_key`** | **`use_key`** |
| Held-out green | **`wait`** | **`wait`** |
| Old-name control | `open` | `open` |
| Class | **Store-works** | **Store-works** |

See [`v17_results.md`](v17_results.md).

## v18 (write `do=` vs write `here=`)

| Check | A write `do=` | B write `here=` |
|-------|---------------|-----------------|
| Untrained | `open` (`action=`) | `open` (`door=`) |
| Trained red | **`use_key`** | **`use_key`** |
| Held-out green | **`wait`** | **`wait`** |
| Old-name control | `open` | `open` |
| Class | **Store-works** | **Store-works** |

See [`v18_results.md`](v18_results.md).

## v19 (shared value-name vs shared place-name)

| Check | A value-name | B place-name |
|-------|--------------|--------------|
| Untrained | `open` (mismatch) | `open` (mismatch) |
| Trained red | **`use_key`** (`do=`) | **`use_key`** (`door=`) |
| Held-out green | **`wait`** | **`wait`** |
| Name control | `open` | `open` |
| Class | **Store-works** | **Store-works** |

See [`v19_results.md`](v19_results.md).

## v20 (find unread W vs find vs junk)

| Check | A find | B find vs junk |
|-------|--------|----------------|
| Untrained | `open` (miss `here=`) | `open` (junk committed, use off) |
| Trained red, unmount W | **`use_key`** (`p99.tag`) | **`use_key`** (not junk) |
| Held-out green | **`wait`** | **`wait`** |
| `door=` control | `open` | `wait` (junk) |
| Class | **Store-works** | **Store-works** |

See [`v20_results.md`](v20_results.md).

## v21 (first-file vs dump-all among W hits)

| Check | A first vs newest | B dump vs newest |
|-------|-------------------|------------------|
| Untrained | `wait` (`aaa.tag`) | `wait` (both files) |
| Trained red, unmount W | **`use_key`** (`p99.tag`) | **`use_key`** (`p99.tag` only) |
| Held-out green | **`wait`** | **`wait`** |
| Recency swap | `wait` | `wait` |
| Class | **Store-works** | **Store-works** |

See [`v21_results.md`](v21_results.md).

## v22 (complete vs stub / joint no clamps)

| Check | A complete vs stub | B joint |
|-------|--------------------|---------|
| Untrained | `open` (stub) | `open` |
| Trained red, unmount W | **`use_key`** (no `when=`) | **`use_key`** |
| Held-out green | **`wait`** | **`wait`** |
| Controls | stub-only `open`; complete-junk `wait` | `door=` `wait`; first `wait`; use-off `open` |
| Class | **Store-works** | **Store-works** |

See [`v22_results.md`](v22_results.md).

## v23 (joint find+complete+use / shared return)

| Check | A split joint | B shared return |
|-------|---------------|-----------------|
| Untrained | `open` (`door=` junk) | `open` |
| Trained red, unmount W | **`use_key`** (no `when=`) | `open` (`junk.tag`) |
| Held-out green | **`wait`** | `open` |
| Train last 50 | 0.84 | **0.00** |
| Class | **Store-works** | **Fail** |

See [`v23_results.md`](v23_results.md).

## TM.0.1.0 (open query names / shared return)

| Check | A split | B shared return |
|-------|---------|-----------------|
| Untrained | `open` (query `action`) | `open` |
| Trained red, unmount W | **`use_key`** (`loc=`) | **`use_key`** (`loc=`) |
| Held-out green | **`wait`** | **`wait`** |
| Train last 50 | 0.88 | 0.90 |
| Class | **Store-works** | **Store-works** |

See [`tm010_results.md`](tm010_results.md).

## TM.0.1.1 (open copy names / shared return)

| Check | A split | B shared return |
|-------|---------|-----------------|
| Untrained | `open` (copy `loc`) | `open` |
| Trained red, unmount W | **`use_key`** (`act=`) | **`use_key`** (`act=`) |
| Held-out green | **`wait`** | **`wait`** |
| Train last 50 | 0.90 | 0.90 |
| Class | **Store-works** | **Store-works** |

See [`tm011_results.md`](tm011_results.md).

## TM.0.1.2 (messy retrieve / shared return)

| Check | A split | B shared return |
|-------|---------|-----------------|
| Untrained | `open` (`p0.tag`) | `open` |
| Trained red, unmount W | **`use_key`** (`where=` `pad=`) | **`use_key`** |
| Held-out green | **`wait`** | **`wait`** |
| Train last 50 | 0.96 | 0.82 |
| Class | **Store-works** | **Store-works** |

See [`tm012_results.md`](tm012_results.md).

## TM.0.2.0 (scale of W / shared return)

| Check | A split | B shared return |
|-------|---------|-----------------|
| W size | 256 | 256 |
| Untrained | `open` (clutter) | `open` |
| Trained red, unmount W | **`use_key`** (`where=` `pad=`) | **`use_key`** |
| Held-out green | **`wait`** | **`wait`** |
| Train last 50 | 0.98 | 0.96 |
| Class | **Store-works** | **Store-works** |

See [`tm020_results.md`](tm020_results.md).

## TM.0.3.0 (a life / shared return)

| Check | A split | B shared return |
|-------|---------|-----------------|
| Free red found messy page | yes (`p99`) | no (`p0`) |
| After ρ reset, W gone | **`use_key`** | `open` |
| Held-out green | **`wait`** | `open` |
| Train last 50 | 0.82 | 0.00 |
| Class | **Store-works** | **Fail** |

See [`tm030_results.md`](tm030_results.md).

## TM.0.3.1 (documents / shared return)

| Check | A split | B shared return |
|-------|---------|-----------------|
| W | all `.md` + prose | same |
| Free red found messy doc | yes (`p99`) | no (`p0`) |
| After ρ reset, W gone | **`use_key`** | `open` |
| Held-out green | **`wait`** | `open` |
| Train last 50 | 0.82 | 0.00 |
| Class | **Store-works** | **Fail** |

See [`tm031_results.md`](tm031_results.md).

## TM.0.3.2 (prose retrieve / shared return)

| Check | A split | B shared return |
|-------|---------|-----------------|
| Filed action=/where= | no | no |
| Free red found prose ints | yes (`n0=0`, `n1=2`) | no |
| After ρ reset, W gone | **`use_key`** | `open` |
| Held-out green | **`wait`** | `open` |
| Train last 50 | 0.56 | 0.00 |
| Class | **Store-works** | **Fail** |

See [`tm032_results.md`](tm032_results.md).

## TM.0.4.0 (channel dial / shared return)

| Check | A split | B shared return |
|-------|---------|-----------------|
| World | channel_dial | channel_dial |
| Free A found place+PRESS | yes (`n0=0`, `n1=1`) | no (clutter) |
| After ρ reset, W gone | **`press`** | `press` (clutter) |
| Held-out C | **`tune`** | `idle` |
| Train last 50 | 0.88 | 0.92 |
| Class | **Store-works** | **Fail** |

See [`tm040_results.md`](tm040_results.md).

## TM.0.5.0 (no answer integers / shared return)

| Check | A split | B shared return |
|-------|---------|-----------------|
| W body integers | none | none |
| Free A found `press` | yes (`w2=press`) | no (clutter) |
| After ρ reset, W gone | **`press`** | `hold` |
| Held-out C | **`tune`** | `hold` |
| Train last 50 | 0.90 | 0.00 |
| Class | **Store-works** | **Fail** |

See [`tm050_results.md`](tm050_results.md).

## TM.0.5.1 (correct a wrong commit / shared return)

| Check | A split | B shared return |
|-------|---------|-----------------|
| n_revised | 11 | 0 |
| After ρ reset, W gone | **`press`** | `hold` |
| Held-out C | **`tune`** | `hold` |
| Revise-off | `hold` | `hold` |
| Train last 50 | 0.60 | 0.00 |
| Class | **Store-works** | **Fail** |

See [`tm051_results.md`](tm051_results.md).

## TM.0.5.2 (unnamed motor / shared return)

| Check | A split | B shared return |
|-------|---------|-----------------|
| W motor names | none | none |
| Free A stamp | yes (`w7=press`) | no (clutter) |
| After ρ reset, W gone | **`press`** | `hold` |
| Held-out C | **`tune`** | `hold` |
| Train last 50 | 0.36 | 0.00 |
| Class | **Store-works** | **Fail** |

See [`tm052_results.md`](tm052_results.md).

## TM.0.5.3 (use-the-fact / shared return)

| Check | A split | B shared return |
|-------|---------|-----------------|
| After A life: A / foil C | **`press` / `hold`** | `hold` / `hold` |
| After C life: C / foil A | **`tune` / `hold`** | `hold` / `hold` |
| Copy-only foil C | **`press`** | `hold` |
| Train last 50 | 0.34 | 0.00 |
| Class | **Store-works** | **Fail** |

See [`tm053_results.md`](tm053_results.md).

## TM.0.5.4 (Open W / shared return)

| Check | A split | B shared return |
|-------|---------|-----------------|
| After A life: A / foil C | **`press` / `hold`** | `hold` / `hold` |
| After C life: C / foil A | **`tune` / `hold`** | `hold` / `hold` |
| Copy-only foil C | **`press`** | `hold` |
| Train last 50 | 0.14 | 0.00 |
| Class | **Store-works** | **Fail** |

See [`tm054_results.md`](tm054_results.md).

## TM.0.5.5 (accumulate S / shared return)

| Check | A split | B shared return |
|-------|---------|-----------------|
| After A life: A / foil C | **`press` / `hold`** | `hold` / `hold` |
| After both lives: A / C | **`press` / `tune`** | `hold` / `hold` |
| Wipe-between: A / C | **`hold` / `tune`** | `hold` / `hold` |
| Copy-only foil C | **`press`** | `hold` |
| Train last 50 | 0.84 | 0.00 |
| Class | **Store-works** | **Fail** |

See [`tm055_results.md`](tm055_results.md).

## TM.0.5.6 (never-wipe train / shared return)

| Check | A split | B shared return |
|-------|---------|-----------------|
| After train, dirty S: A / foil C | **`press` / `hold`** | `hold` / `hold` |
| C life on dirty S: A / C | **`press` / `tune`** | `hold` / `hold` |
| Wipe-between: A / C | **`hold` / `hold`** | `hold` / `hold` |
| Train last 50 | 0.90 | 0.00 |
| Class | **Store-works** | **Fail** |

See [`tm056_results.md`](tm056_results.md).

## TM.0.5.7 (find without unique rare / shared return)

| Check | A split | B shared return |
|-------|---------|-----------------|
| After train, dirty S: A / foil C | **`press` / `hold`** | `press` / `hold` |
| C life on dirty S: A / C | **`press` / `tune`** | `press` / `hold` |
| Wipe-between: A / C | **`hold` / `hold`** | `hold` / `hold` |
| Rare clutter pages | 3 | 3 |
| Train last 50 | 0.92 | 0.86 |
| Class | **Store-works** | **Fail** |

See [`tm057_results.md`](tm057_results.md).

## TM.0.5.8 (scale of Open W / shared return)

| Check | A split | B shared return |
|-------|---------|-----------------|
| After train, dirty S: A / foil C | **`press` / `hold`** | **`press` / `hold`** |
| C life on dirty S: A / C | **`press` / `tune`** | **`press` / `tune`** |
| Wipe-between: A / C | **`hold` / `tune`** | `hold` / `hold` |
| Distinct clutter | 64 | 64 |
| Train last 50 | 0.90 | 0.94 |
| Class | **Store-works** | **Store-works** |

See [`tm058_results.md`](tm058_results.md).

## TM.0.5.9 (correct dirty S / shared return)

| Check | A split | B shared return |
|-------|---------|-----------------|
| After train, S: A / foil C | **`press` / `hold`** | `press` / `hold` |
| C life on that S: A / C | **`press` / `tune`** | `press` / `hold` |
| Wipe-between: A / C | **`hold` / `tune`** | `hold` / `hold` |
| Train S n files | **1** | **1** |
| Train last 50 | 0.92 | 0.98 |
| Class | **Store-works** | **Fail** |

See [`tm059_results.md`](tm059_results.md).

## TM.0.6.0 (English life / shared return)

| Check | A split | B shared return |
|-------|---------|-----------------|
| After A life: A / foil C | **`press` / `hold`** | **`press` / `hold`** |
| After C life: C / foil A | **`tune` / `hold`** | **`tune` / `hold`** |
| Bind-off A | **`hold`** | **`hold`** |
| Train last 50 | 0.88 | 0.46 |
| Class | **Store-works** | **Store-works** |

See [`tm060_results.md`](tm060_results.md).

## TM.0.6.1 (one bind / shared return)

| Check | A split | B shared return |
|-------|---------|-----------------|
| After A life: A / foil C | **`press` / `hold`** | **`press` / `hold`** |
| After C life: C / foil A | **`tune` / `hold`** | **`tune` / `hold`** |
| Nonce-only A | **`hold`** | **`hold`** |
| Bind-all nonce A | **`press`** | **`press`** |
| Train last 50 | 0.88 | 0.90 |
| Class | **Store-works** | **Store-works** |

See [`tm061_results.md`](tm061_results.md).

## TM.0.6.2 (never-wipe English / shared return)

| Check | A split | B shared return |
|-------|---------|-----------------|
| After train, dirty S: A / foil C | **`press` / `hold`** | **`press` / `hold`** |
| C life on dirty S: A / C | **`press` / `hold`** | **`press` / `tune`** |
| Nonce-only A | **`hold`** | **`hold`** |
| Bind-all nonce A | **`press`** | **`press`** |
| Train last 50 | 0.92 | 0.92 |
| Class | **Fail** | **Store-works** |

See [`tm062_results.md`](tm062_results.md).

## TM.0.6.3 (new-here stamp / shared return)

| Check | A split | B shared return |
|-------|---------|-----------------|
| After train, dirty S: A / foil C | **`press` / `hold`** | **`press` / `hold`** |
| C life on dirty S: A / C | **`press` / `tune`** | **`press` / `tune`** |
| Nonce-only A | **`hold`** | **`hold`** |
| Bind-all nonce A | **`press`** | **`press`** |
| Train last 50 | 0.92 | 0.92 |
| Class | **Store-works** | **Store-works** |

See [`tm063_results.md`](tm063_results.md).

## TM.0.6.4 (English find without unique rare / shared return)

| Check | A split | B shared return |
|-------|---------|-----------------|
| After train, dirty S: A / foil C | `press` / `hold` | **`press` / `hold`** |
| C life on dirty S: A / C | `press` / `tune` | **`press` / `tune`** |
| Train S binds | `push` **and** xenon/neon/krypton | same |
| Nonce-only A | **`hold`** | **`hold`** |
| Train last 50 | 0.94 | 0.94 |
| Class | **Fail** | **Store-works** |

See [`tm064_results.md`](tm064_results.md).

## TM.0.6.5 (concurrent bind / shared return)

| Check | A split | B shared return |
|-------|---------|-----------------|
| After train, dirty S: A / foil C | **`press` / `hold`** | `press` / `hold` |
| C life on dirty S: A / C | **`press` / `tune`** | `press` / **`hold`** |
| Train S binds | **`push` only** | `neon` |
| Train last 50 | 0.92 | **0.96** |
| Class | **Store-works** | **Fail** |

See [`tm065_results.md`](tm065_results.md).

## TM.0.6.6 (correct dirty English S / shared return)

| Check | A split | B shared return |
|-------|---------|-----------------|
| After train, S: A / foil C | **`press` / `hold`** | **`press` / `hold`** |
| C life on that S: A / C | `press` / `tune` | **`press` / `tune`** |
| Train S n files | **1** | **1** |
| Train S binds | **`push`** | `krypton` |
| C life binds | `push` + **`xenon`** | `krypton` + `xenon` |
| Train last 50 | 1.00 | 1.00 |
| Class | **Fail** | **Store-works** |

See [`tm066_results.md`](tm066_results.md).

## TM.0.6.7 (in-hand new-here / shared return)

| Check | A split | B shared return |
|-------|---------|-----------------|
| After train, S: A / foil C | **`press` / `hold`** | **`press` / `hold`** |
| C life on that S: A / C | `press` / `tune` | **`press` / `tune`** |
| Train S n files | **1** | **1** |
| Train S binds | **`push`** | `krypton` |
| C life binds | `push` + **`neon`** | `krypton` + `neon` |
| Train last 50 | 1.00 | 1.00 |
| Class | **Fail** | **Store-works** |

See [`tm067_results.md`](tm067_results.md).

## TM.0.6.8 (find-novel / shared return)

| Check | A split | B shared return |
|-------|---------|-----------------|
| After train, S: A / foil C | **`press` / `hold`** | **`press` / `hold`** |
| C life on that S: A / C | **`press` / `tune`** | **`press` / `tune`** |
| Train S n files | **1** | **1** |
| Train S binds | **`push`** | **`push`** |
| C life binds | **`push` + `adjust`** | **`push` + `adjust`** |
| Train last 50 | 1.00 | 0.94 |
| Class | **Store-works** | **Store-works** |

See [`tm068_results.md`](tm068_results.md).

## TM.0.6.9 (find-novel without unique two-rare / shared return)

| Check | A split | B shared return |
|-------|---------|-----------------|
| After train, S: A / foil C | **`press` / `hold`** | **`press` / `hold`** |
| C life on that S: A / C | `press` / `tune` | **`press` / `tune`** |
| Train S n files | **1** | **1** |
| Train S binds | **`neon`** | `xenon` |
| C life binds | `neon` + **`xenon`** | `xenon` + `neon` |
| Train last 50 | 0.94 | 0.92 |
| Class | **Fail** | **Store-works** |

See [`tm069_results.md`](tm069_results.md).

