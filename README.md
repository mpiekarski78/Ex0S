# Three-memory experimental model

> Sibling of the BDH experience-driven state work. **Not** a fork of Pathway BDH.  
> BDH baseline (Category B — short-term adaptive memory): [mpiekarski78/bdh](https://github.com/mpiekarski78/bdh) · [conclusion](https://github.com/mpiekarski78/bdh/blob/main/docs/conclusion.md)

## Question

> Can frozen innate drives + learning rules fill an **inspectable** world-knowledge store from experience, such that facts **survive reset of the working trace** — while the trace alone does not?

Biology’s lesson here: hardcode **drives and learning rules**, leave **world-knowledge** to experience, and do **not** confuse a short trace with a life of knowledge.

## Result (v0): **Store-works**

Full write-up: [`docs/conclusion.md`](docs/conclusion.md). Comparison: [`docs/comparison_bdh.md`](docs/comparison_bdh.md).

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

## Three pieces

| Piece | Role | Survives ρ reset? |
|-------|------|-------------------|
| Frozen cortex | Species prior (sensors / dynamics) | yes (fixed weights) |
| Working trace ρ | Session residue | **no** |
| World store S | Inspectable life-of-knowledge (JSON) | **yes** |

## Status

| Phase | Status | Notes |
|-------|--------|-------|
| Public repo | done | this repository |
| v0 key/door | **Store-works** | [`docs/conclusion.md`](docs/conclusion.md) |
| Compare to BDH Category B | done | [`docs/comparison_bdh.md`](docs/comparison_bdh.md) |
| v1 tiny LM | planned (gated) | [`docs/v1_plan.md`](docs/v1_plan.md) — not started |

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python tests/test_smoke.py
python -m experiments.run_v0
```

Protocol: [`docs/protocol.md`](docs/protocol.md).

## Layout

```text
three_memory/     # cortex, ρ, store S, drives, agent, env
experiments/      # run_v0 CLI
docs/             # protocol, comparison, conclusion, v1 plan
tests/            # smoke tests
runs/             # gitignored metrics
```

## What this is not

- Not a Pathway BDH patch or PR
- Not hardcoded “survive / reproduce” objectives
- Not a chatbot / agent product
- Not Category D on ρ — ρ stays session-only
