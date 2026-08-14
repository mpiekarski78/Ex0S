# Three-memory (TM)

**Current:** TM.0.3.0. Toy series: v0–v23.

> Sibling of the BDH experience-driven state work. **Not** a fork of Pathway BDH.  
> BDH baseline (Category B — short-term adaptive memory): [mpiekarski78/bdh](https://github.com/mpiekarski78/bdh) · [conclusion](https://github.com/mpiekarski78/bdh/blob/main/docs/conclusion.md)

BDH showed that a working trace ρ is useful in-session and gone after reset. This repo does not try to make ρ long-term. It asks what belongs **beside** ρ.

## Question

**v0 (answered):** frozen drives can write an inspectable fact into S; after ρ reset the fact still steers behavior **if and only if** it lives in S. disable-S is BDH Category B again. See [`docs/conclusion.md`](docs/conclusion.md).

**Now:** DNA is a **recipe for the machine**, not Wikipedia. English is not a species prior. Hardcode **drives and learning rules**. Leave **world-knowledge to experience**. Facts go in files you can open. Skills (when to write, whether to copy, which field to match or emit) go in **boxed heads**. The cortex SHA256 must not move.

> Can a frozen cortex plus boxed learning rules fill an inspectable store from a free life, and learn to use those files — without putting facts into genome weights, and without wiring the answer as English or as a USE_KEY/WAIT table?

Honest status after TM.0.3.0: the **split is real on a toy that can find/commit in a free life and still use S after ρ reset**. Not a general learner. Shared return on that life **Fail**s (last-50 0). Query/copy names and messy search worked in earlier TM runs. Still genome: generic `logits[int] += 3.0`, `{has_code, has_rare}`, frozen commit-on-hit, four discrete acts. Tiny LSTM still needs a taught tool grammar for English. W is `.tag` files, not Wikipedia. No code, no cameras.

## Result (v0–v23 toy, TM.0.x)

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
v12 learn select vs dump: [`docs/v12_results.md`](docs/v12_results.md).  
v13 copy `action=` from the file: [`docs/v13_results.md`](docs/v13_results.md).  
v14 pick-one vs write schema: [`docs/v14_results.md`](docs/v14_results.md).  
v15 joint, no clamps: [`docs/v15_results.md`](docs/v15_results.md).  
v16 ok= vs newest / shared return: [`docs/v16_results.md`](docs/v16_results.md).  
v17 read `do=` vs match `here=`: [`docs/v17_results.md`](docs/v17_results.md).  
v18 write `do=` vs write `here=`: [`docs/v18_results.md`](docs/v18_results.md).  
v19 shared name: [`docs/v19_results.md`](docs/v19_results.md).  
v20 find in W: [`docs/v20_results.md`](docs/v20_results.md).  
v21 select among W hits: [`docs/v21_results.md`](docs/v21_results.md).  
v22 complete vs stub / joint: [`docs/v22_results.md`](docs/v22_results.md).  
v23 joint wiki / shared return: [`docs/v23_results.md`](docs/v23_results.md).  
TM.0.1.0 open query names: [`docs/tm010_results.md`](docs/tm010_results.md).  
TM.0.1.1 open copy names: [`docs/tm011_results.md`](docs/tm011_results.md).  
TM.0.1.2 messy retrieve: [`docs/tm012_results.md`](docs/tm012_results.md).  
TM.0.2.0 scale of W: [`docs/tm020_results.md`](docs/tm020_results.md).  
TM.0.3.0 a life: [`docs/tm030_results.md`](docs/tm030_results.md).  
Comparison: [`docs/comparison_bdh.md`](docs/comparison_bdh.md).

| Check | Outcome |
|-------|---------|
| Frozen cortex SHA256 after a life | unchanged |
| Boxed policy after a life (v8+) | **may move** (when/whether/which-name; not facts) |
| A learns `red door opens only with key` into S (v0) | yes (plain JSON) |
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
| v12 retrieve head, held-out blue | **Store-works** (untrained dump `wait`; trained select; blue `open`) |
| v13 use-gate, generic copy | **Store-works** (untrained ignores planted tag; red `use_key`; green `wait`) |
| v14 A pick-one / B schema | **Store-works** / **Store-works** (newest match; include `action=` in the note) |
| v15 joint, no clamps | **Store-works** (write+schema+use+pick; red `use_key`; green `wait`) |
| v16 A ok= vs newest / B shared return | **Store-works** / **Fail** (prefer `ok=1`; one return stays `open`) |
| v17 A do= / B here= | **Store-works** / **Store-works** (learn field name to copy or match) |
| v18 A write do= / B write here= | **Store-works** / **Store-works** (learn field name to emit) |
| v19 A shared value-name / B shared place-name | **Store-works** / **Store-works** (write and read learn a convention) |
| v20 A find unread W / B find vs junk | **Store-works** / **Store-works** (query `here=`; commit `p99.tag`; junk on `door=` does not leak `use_key`) |
| v21 A first-file / B dump-all among W hits | **Store-works** / **Store-works** (newest `when=` over filename-first or dump; recency-swap stays `wait`) |
| v22 A complete vs stub / B joint no clamps | **Store-works** / **Store-works** (payload over stub, no `when=`; find+pick+use together) |
| v23 A joint find+complete+use / B shared return | **Store-works** / **Fail** (no `when=`; split credit load-bearing; shared return last-50 0) |
| TM.0.1.0 A open query names / B shared return | **Store-works** / **Store-works** (keys from files, not `{door, here}`; two-head shared return works) |
| TM.0.1.1 A open copy names / B shared return | **Store-works** / **Store-works** (keys from the hit, not `{action, do}`; green must not copy `loc=2`) |
| TM.0.1.2 A messy retrieve / B shared return | **Store-works** / **Store-works** (rank files; no exact `loc=`/`door=`; extra `pad=`) |
| TM.0.2.0 A scale of W / B shared return | **Store-works** / **Store-works** (256 unread files; same `{has_code, has_rare}`; no shrink) |
| TM.0.3.0 A free life / B shared return | **Store-works** / **Fail** (find/commit in life; shared return last-50 0) |

## Five pieces

| Piece | Role | Survives ρ reset? |
|-------|------|-------------------|
| Frozen cortex | Species prior (sensors/dynamics). No facts. SHA256 fixed. | yes |
| Boxed policy | Learning rules (when / whether / which name). May move. Not Wikipedia. | yes (weights ≠ facts) |
| Working trace ρ | Session residue | **no** |
| World store S | Committed life (inspectable notes) | **yes** |
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
| v12 learn select vs dump | **Store-works** | retrieve head; held-out blue; [`docs/v12_results.md`](docs/v12_results.md) |
| v13 copy action= | **Store-works** | use-gate; no USE_KEY/WAIT table; [`docs/v13_results.md`](docs/v13_results.md) |
| v14 pick vs schema | **Store-works** / **Store-works** | one-of-N matches vs complete note; [`docs/v14_results.md`](docs/v14_results.md) |
| v15 joint no clamps | **Store-works** | four heads together; [`docs/v15_results.md`](docs/v15_results.md) |
| v16 ok= vs newest / shared return | **Store-works** / **Fail** | recency vs `ok=1`; split credit was load-bearing; [`docs/v16_results.md`](docs/v16_results.md) |
| v17 do= / here= | **Store-works** / **Store-works** | copy `do=` or match `here=`; [`docs/v17_results.md`](docs/v17_results.md) |
| v18 write do= / write here= | **Store-works** / **Store-works** | emit `do=` or `here=`; [`docs/v18_results.md`](docs/v18_results.md) |
| v19 shared name | **Store-works** / **Store-works** | convention `do=` / `door=`; [`docs/v19_results.md`](docs/v19_results.md) |
| v20 find in W | **Store-works** / **Store-works** | unread `p99.tag` vs `door=` junk; [`docs/v20_results.md`](docs/v20_results.md) |
| v21 select among W hits | **Store-works** / **Store-works** | newest vs first/dump; [`docs/v21_results.md`](docs/v21_results.md) |
| v22 complete vs stub / joint | **Store-works** / **Store-works** | no `when=` cheat; three heads together; [`docs/v22_results.md`](docs/v22_results.md) |
| v23 joint wiki / shared return | **Store-works** / **Fail** | find+complete+use, no `when=`; shared return starves; [`docs/v23_results.md`](docs/v23_results.md) |
| TM.0.1.0 open query names | **Store-works** / **Store-works** | files supply query keys; `{door, here}` menu off; [`docs/tm010_results.md`](docs/tm010_results.md) |
| TM.0.1.1 open copy names | **Store-works** / **Store-works** | files supply copy keys; `{action, do}` menu off; [`docs/tm011_results.md`](docs/tm011_results.md) |
| TM.0.1.2 messy retrieve | **Store-works** / **Store-works** | rank unread files; exact match misses; [`docs/tm012_results.md`](docs/tm012_results.md) |
| TM.0.2.0 scale of W | **Store-works** / **Store-works** | 256 messy files; same search; [`docs/tm020_results.md`](docs/tm020_results.md) |
| TM.0.3.0 a life | **Store-works** / **Fail** | free life find/commit; shared return starves; [`docs/tm030_results.md`](docs/tm030_results.md) |

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
python tests/test_v12.py
python tests/test_v13.py
python tests/test_v14.py
python tests/test_v15.py
python tests/test_v16.py
python tests/test_v17.py
python tests/test_v18.py
python tests/test_v19.py
python tests/test_v20.py
python tests/test_v21.py
python tests/test_v22.py
python tests/test_v23.py
python tests/test_tm010.py
python tests/test_tm011.py
python tests/test_tm012.py
python tests/test_tm020.py
python tests/test_tm030.py
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
python -m experiments.run_v12
python -m experiments.run_v13
python -m experiments.run_v14
python -m experiments.run_v15
python -m experiments.run_v16
python -m experiments.run_v17
python -m experiments.run_v18
python -m experiments.run_v19
python -m experiments.run_v20
python -m experiments.run_v21
python -m experiments.run_v22
python -m experiments.run_v23
python -m experiments.run_tm010
python -m experiments.run_tm011
python -m experiments.run_tm012
python -m experiments.run_tm020
python -m experiments.run_tm030
```

Protocol: [`docs/protocol.md`](docs/protocol.md).

## Layout

```text
three_memory/     # cortex, ρ, S, W library, drives, agent, env, byte LM
experiments/      # run_v0 … run_v23, run_tm010 … run_tm030, train_prior
docs/             # protocol, comparison, conclusion, v1–v23 and TM.0.x results
tests/
runs/             # gitignored
checkpoints/      # gitignored (prior.pt)
```

## What this is not

- Not a Pathway BDH patch or PR
- Not Category D on ρ — ρ stays session-only
- Not Wikipedia in the genome, and not English as a species prior
- Not a general learner / chatbot / agent product
- Not RAG (v3 is string-matched `.md` files, no embeddings)
- Not hardcoded “survive / reproduce” objectives
