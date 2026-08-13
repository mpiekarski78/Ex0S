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
