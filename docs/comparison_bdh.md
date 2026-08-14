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
