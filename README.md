# Three-memory experimental model

> Sibling of the BDH experience-driven state work. **Not** a fork of Pathway BDH.  
> BDH baseline (Category B — short-term adaptive memory): [mpiekarski78/bdh](https://github.com/mpiekarski78/bdh) · [conclusion](https://github.com/mpiekarski78/bdh/blob/main/docs/conclusion.md)

## Question

> Can frozen innate drives + learning rules fill an **inspectable** world-knowledge store from experience, such that facts **survive reset of the working trace** — while the trace alone does not?

Biology’s lesson here: hardcode **drives and learning rules**, leave **world-knowledge** to experience, and do **not** confuse a short trace with a life of knowledge.

## Result (v0–v11)

v0: [`docs/conclusion.md`](docs/conclusion.md).  
v1 NOTE-copy: [`docs/v1_results.md`](docs/v1_results.md).  
v2 raw retrieve: [`docs/v2_results.md`](docs/v2_results.md).  
v3 markdown files (no RAG): [`docs/v3_results.md`](docs/v3_results.md).  
v4 select among notes: [`docs/v4_results.md`](docs/v4_results.md).  
v5 collect from unread W: [`docs/v5_results.md`](docs/v5_results.md).  
v6 use-skill on plain prior: [`docs/v6_results.md`](docs/v6_results.md).  
v7 native tags (no English): [`docs/v7_results.md`](docs/v7_results.md).  
v8 boxed use-policy (cortex frozen): [`docs/v8_results.md`](docs/v8_results.md).  
v9 write from a life (W has no answer): [`docs/v9_results.md`](docs/v9_results.md).  
v10 free life (no forced curriculum): [`docs/v10_results.md`](docs/v10_results.md).  
v11 select among authored notes: [`docs/v11_results.md`](docs/v11_results.md).  
Comparison: [`docs/comparison_bdh.md`](docs/comparison_bdh.md).

| Check | Outcome |
|-------|---------|
| Same frozen cortex (species prior) | yes |
| Weights after experience | unchanged (SHA256) |
| A learns `red door opens only with key` into S | yes (plain JSON) |
| A correct after ρ reset (S kept) | yes |
| B (foil) after ρ reset | no |
| disable-S: correct before ρ reset | yes (session residue) |
| disable-S: correct after ρ reset | **no** (BDH-like Category B) |
| Reset S | effect gone |
| v1 `my lo` after ρ reset, S on | P(`v`)=0.988 (taught NOTE-copy) |
| v1 same, S off | P(`v`)=0.027 (empty prior) |
| v2 raw retrieve after ρ reset, S on | P(`v`)=0.093 ≈ prior (**Trace-only**) |
| v3 new agent, load `.md` only (NOTE prior) | P(`v`)=0.988 (**Store-works**; JS vs in-process = 0) |
| v3 new agent, load `.md` only (plain prior) | P(`v`)=0.093 (**Trace-only**) |
| v4 select 1 of 13 notes (NOTE prior) | P(`v`)=0.988; dump-all P(`v`)=0.007 |
| v5 commit W→S then unmount W (NOTE prior) | P(`v`)=0.988; peek then unmount → prior |
| v6 tool grammar, plain prior, no `love` in window | P(`v`)=0.649 (**Store-works**) |
| v6 fewshot / untaught NOTE, plain prior | P(`v`)≈0.053 (**Fail**) |
| v7 native tags, no English prior | **Store-works** (`use_key` after reload/collect; dump-all/`peek` → `open`) |
| v8 boxed policy, held-out green | **Store-works** (red unmount `use_key`; green `wait`; cortex hash unchanged) |
| v9 write from events, no answer in W | **Store-works** (authors `d0.tag`/`d2.tag`; red `use_key`; green `wait`) |
| v10 free life, n_forced=0 | **Store-works** (red not the v9 script; green found WAIT; greedy probe after ρ reset) |
| v11 two lives, select vs dump | **Store-works** (select red `use_key` / green `wait`; dump-all red `wait`) |

## Four pieces

| Piece | Role | Survives ρ reset? |
|-------|------|-------------------|
| Frozen cortex | Species prior + use/collect rules | yes (fixed weights) |
| Working trace ρ | Session residue | **no** |
| World store S | Life-of-knowledge (committed notes) | **yes** |
| Library W | Unread available data | **no** (not owned until commit) |

## Status

| Phase | Status | Notes |
|-------|--------|-------|
| Public repo | done | this repository |
| v0 key/door | **Store-works** | [`docs/conclusion.md`](docs/conclusion.md) |
| Compare to BDH Category B | done | [`docs/comparison_bdh.md`](docs/comparison_bdh.md) |
| v1 tiny LM | **Store-works** | taught NOTE-copy; [`docs/v1_results.md`](docs/v1_results.md) |
| v2 raw retrieve | **Trace-only** | no NOTE-copy; [`docs/v2_results.md`](docs/v2_results.md) |
| v3 markdown S | **Store-works** / **Trace-only** | files on disk, no RAG; [`docs/v3_results.md`](docs/v3_results.md) |
| v4 select | **Store-works** / **Fail** | pick one `.md` among 13; [`docs/v4_results.md`](docs/v4_results.md) |
| v5 collect | **Store-works** / **Fail** | W→S commit vs peek; [`docs/v5_results.md`](docs/v5_results.md) |
| v6 use-skill | **Store-works** / **Fail** | tool vs fewshot on plain prior; [`docs/v6_results.md`](docs/v6_results.md) |
| v7 native tags | **Store-works** | bits + integer `.tag` files; [`docs/v7_results.md`](docs/v7_results.md) |
| v8 boxed use-policy | **Store-works** | policy learns when to commit/apply; cortex frozen; [`docs/v8_results.md`](docs/v8_results.md) |
| v9 write from a life | **Store-works** | author S from events; W has no answer; [`docs/v9_results.md`](docs/v9_results.md) |
| v10 free life | **Store-works** | no forced curriculum; live then write; [`docs/v10_results.md`](docs/v10_results.md) |
| v11 select authored notes | **Store-works** | two lives, pick the match; dump-all mixes; [`docs/v11_results.md`](docs/v11_results.md) |

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python tests/test_smoke.py
python tests/test_v1_smoke.py
python tests/test_md_store.py
python tests/test_library.py
python tests/test_v6.py
python tests/test_v7.py
python tests/test_v8.py
python tests/test_v9.py
python tests/test_v10.py
python tests/test_v11.py
python -m experiments.run_v0
python -m experiments.train_prior
python -m experiments.run_v1
python -m experiments.train_prior --plain
python -m experiments.run_v2
python -m experiments.run_v3 --both
python -m experiments.run_v4 --both
python -m experiments.run_v5 --both
python -m experiments.run_v6 --all-modes
python -m experiments.run_v7
python -m experiments.run_v8
python -m experiments.run_v9
python -m experiments.run_v10
python -m experiments.run_v11
```

Protocol: [`docs/protocol.md`](docs/protocol.md).

## Layout

```text
three_memory/     # cortex, ρ, S, W library, drives, agent, env, byte LM
experiments/      # run_v0 … run_v11, train_prior
docs/             # protocol, comparison, conclusion, v1–v11 results
tests/
runs/             # gitignored
checkpoints/      # gitignored (prior.pt)
```

## What this is not

- Not a Pathway BDH patch or PR
- Not hardcoded “survive / reproduce” objectives
- Not a chatbot / agent product
- Not Category D on ρ — ρ stays session-only
- Not RAG (v3 is string-matched `.md` files, no embeddings)
