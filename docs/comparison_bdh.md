# Comparison to BDH Category B

BDH is the **trace-only baseline**, not a scoreboard.  
Source: [mpiekarski78/bdh](https://github.com/mpiekarski78/bdh) · [docs/conclusion.md](https://github.com/mpiekarski78/bdh/blob/main/docs/conclusion.md)

v0 compares **the same questions**. Door success rate is not JS on `my lo`.

## Shared checklist (reported v0 run)

| Check | BDH (measured) | three-memory S **off** | three-memory S **on** |
|-------|----------------|------------------------|------------------------|
| Slow weights after experience | unchanged | unchanged | unchanged |
| Working trace diverges / moves | yes (ρ L2 ≫ 0) | yes (session success action + ρ) | yes |
| Same probe, different behavior | yes (fragile) | yes before ρ reset (`use_key`) | yes |
| Reset ρ, keep everything else | effect gone | effect gone (`open`) | **effect remains** (`use_key`) |
| Reset S (or never write S) | n/a | n/a | effect gone |
| Snapshot restore of ρ | exact | action match after restore | action match (knowledge does not *need* it) |
| Inspectable record of the fact | no | no | **yes** (`store_A.json`) |

## Classification

| System | Letter / label |
|--------|----------------|
| Public BDH | **Category B** — short-term adaptive memory |
| three-memory, S off | **Trace-only** (same idea as BDH B): works in-session, dies on ρ reset |
| three-memory, S on | **Store-works**: fact survives ρ reset and is readable in JSON |

## Honest limits

- Different architecture and task (key/door vs Shakespeare bytes).
- BDH was not given an explicit store; winning Store-works does **not** mean public BDH was wrong.
- It means the missing box for a *life of knowledge* was an inspectable store, not a longer ρ.
- v0 retrieve is **tag→action rules**, not reading English from S. The JSON `what` field is for inspection.
- Comparable *numbers* (JS, ΔP, `my lord`/`my love`) wait for **v1** language probes if pursued.
