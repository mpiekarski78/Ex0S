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

