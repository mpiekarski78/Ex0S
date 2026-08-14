# v10 results: free life (no forced curriculum)

**Date:** 14 August 2026  
**Classification:** **Store-works**  
**Run:** `runs/2026-08-14_085858_v10`

## Question

> Can the agent **live** a door-opening (no OPEN→PICK_KEY→USE_KEY script), author the note, and still use it after ρ reset — cortex frozen, W with no answer?

v9 puppets the life. v10: ε-greedy over **percept affordances** during the life (`explore=True`). The probe is greedy (`explore=False`). Write box and `{here, that act}` template are unchanged. Collect off. W is clutter only.

## Predeclared

| ID | If |
|----|----|
| Confound | Cortex hash moves; disable-S still `use_key`; answer in W; `n_forced > 0`; probe used exploration |
| Fail | Free red never opens; no authored file; greedy probe fails; free green never finds WAIT |
| Store-works | `n_forced = 0`; S authored from a real opening; red `use_key` and green `wait` after ρ reset; empty S / disable-S fail |

Affordances are frozen percept rules: cannot USE_KEY without holding a key; PICK_KEY only if the key is visible. That is not “red → use_key.”

## Headline

| Check | Result |
|-------|--------|
| n_forced (untrained / red / green) | **0 / 0 / 0** |
| W has `d0.tag` / `d2.tag` | no / no |
| Untrained greedy probe after ρ reset | `open` |
| Red free life | opened; authored `d0.tag` (`action=2`) |
| Red life actions | `pick_key, open, open, wait, open, open, wait, open, open, use_key` |
| Red greedy probe after ρ reset | **`use_key`** |
| Empty S / disable-S | `open` |
| Green free life | opened; authored `d2.tag` (`action=0`) |
| Green life actions | `wait` |
| Green greedy probe after ρ reset | **`wait`** |
| Cortex SHA256 | unchanged |
| Policy SHA256 | changed |
| Train return last 50 | 0.84 |

The red sequence is **not** the v9 script. Green opened on a sampled WAIT (affordances `{wait, open}`), then the greedy probe still WAIT after ρ reset because of the file, not because the probe was exploring.

## What this means

v8: take a match.  
v9: write when shown.  
v10: **live, then write.**

Facts stay in files. Cortex stays frozen. The write skill transferred to a free green life.

## Honest limits

- Exploration is ε-greedy over a tiny affordance set, not a general curiosity learner.
- Green can open on the first WAIT by chance; the test is that we did not step a WAIT tuple, and the **probe** is greedy.
- Last-50 return 0.84: some train lives still fail to open or write. Not a perfect creature.
- Apply/retrieve is still the frozen tag→logit grammar.

## Reproduce

```bash
python tests/test_v10.py
python tests/test_v9.py
python -m experiments.run_v10
```
