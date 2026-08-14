# v9 results: write from a life (W has no answer)

**Date:** 14 August 2026  
**Classification:** **Store-works**  
**Run:** `runs/2026-08-14_000455_v9`

## Question

> Can a boxed policy learn *when* to author a note from a door-opening, while W contains no answer and the frozen cortex never stores the fact?

v8 copied a labeled `.tag` from W. That is collect. v9 **writes** `{door, action}` from the act that opened the door. Collect is off. W is clutter only — no `d0.tag`, no `d2.tag`.

## Predeclared

| ID | If |
|----|----|
| Confound | Cortex hash moves; disable-S after ρ reset still `use_key`; or the answer file was in W |
| Fail | Policy hash unchanged; no file authored; red dies after ρ reset; held-out green life fails |
| Store-works | Cortex unchanged, policy changed, S authored from events, red and green work after ρ reset, empty S / disable-S fail |

Held-out: green life (`experience_green`: OPEN fails, WAIT opens). Same trained write-policy, no green file in W. Frozen WHAT is generic: on `opened`, write `{door: here, action: the act}`. Policy features: `{s_hit, opportunity}` only — no door id.

## Headline

Untrained write gate starts off (`b_write = -1.2`). After a red life with no write: **`open`**.

| Check | Action / files | Correct? |
|-------|----------------|----------|
| Untrained red, ρ reset | open, S empty | no (wanted) |
| W contains `d0.tag` / `d2.tag` | no / no | — |
| Red life authors | **`d0.tag`** `door=0 action=2` | yes |
| Red after ρ reset, W unmounted | **`use_key`** | **yes** |
| Empty S | open | no |
| disable-S after ρ reset | open | no |
| Held-out green life authors | **`d2.tag`** `door=2 action=0` | yes |
| Green after ρ reset, W unmounted | **`wait`** | **yes** |
| Cortex SHA256 | unchanged | yes |
| Policy SHA256 | changed | yes |
| Train return last 50 | 0.96 | — |

Authored red file:

```text
# d0
action=2
door=0
```

Authored green file:

```text
# d2
action=0
door=2
```

Green `wait` is what the **life** wrote, not a library cheat and not “that door” in the policy.

## What trained

Each episode: forced red curriculum (OPEN, PICK_KEY, USE_KEY) so a door-opening fires; policy chooses whether to write; unmount W; reset ρ; probe. Reward on the probe.

The genome does not contain “red → use_key”. It contains: **when a door opens, a note may record here + that act**. The policy learns **when**.

## What this means

v8: take a match from W.  
v9: a life becomes a note in S.

Facts still live in files. Cortex stays frozen (seed 1337). Held-out green is the check that the write skill transferred.

## Honest limits

- Forced curriculum so the opening event exists; `n_forced` is the whole life (3 red steps / 2 green). Not free exploration.
- Linear REINFORCE, two features, 400 red episodes.
- Apply/retrieve is still the frozen tag→logit grammar.
- W clutter is unused (collect off). Peek vs commit of library files is v8, not this test.

## Reproduce

```bash
python tests/test_v9.py
python tests/test_v8.py
python -m experiments.run_v9
```
