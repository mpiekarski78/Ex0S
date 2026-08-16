# TM.0.16.RELATE — candidate relations under ambiguity

**Lab:** TM.0.16.RELATE (bookkeeping, not a product stamp)  
**Product under test:** Ex0S **0.0.004**  
**Recorded:** `python -m experiments.run_tm016relate --write-lock` → **16/16**  
**Flags:** `earned_next=false`, `ex0s=null`

## Claim

> A frozen developmental recipe can accumulate candidate relations across repeated ambiguous multi-symbol event streams and use converging evidence to select an invariant relational route for later composition, while surface distractor relations remain inspectable competing hypotheses, without the apparatus choosing which visible-symbol transitions should control behavior.

## Mechanism

- ABI: `observe_event({"visible": [...], "focus": ...})` — authoring reads **`visible` only**; **MUST NOT read `focus`**
- Candidate rule: all-pairs `prev_visible × curr_visible` → `experience_skel` (incl. self-pairs); normalize strip/lower/dedupe/sort; skip motors; **no prune**
- Episode boundary: `end_event_episode()` clears only `_rel_prev_visible` (not S/ρ/κ); lives accumulate without requiring `reset_rho`
- Compose selects the unique `(support, -contradiction)` winner; **HOLD on ties**; losers remain inspectable in S

## Battery

| Cell | Result |
|------|--------|
| D0 birth unreachable | OK |
| D1 one ambiguous exposure → HOLD | OK |
| D2 varying clutter → X→A winner; losers in S | OK — losers `x→q/r/u` remain |
| D3 full route X→A→Y + PRESS | OK |
| D4 surface invariant (same latent) | OK |
| D5 counterfactual X→B→Y | OK |
| D6 reset_ρ | OK |
| D7 newborn reload | OK |
| D8 dual-strip | OK |
| D9 focus not relation oracle | OK — XAY/QRS/random identical; focus-only path blocked |
| D10 irreducible equal evidence → HOLD | OK |
| D11 episode boundary (no Y→X seam) | OK |
| D12 fid / storage order | OK |
| D13 channel + oracle | OK |
| D14 weights / no shortcut | OK |
| D15 nasty (normalize / stale / provenance) | OK — includes legal N=1 `{X}→{A}` |

## Locks

- Prereg (before coding): [`relate_016.prereg.lock`](relate_016.prereg.lock)
- Freeze: [`genome_016.lock`](genome_016.lock) + [`relate_016.lock`](relate_016.lock)
- Prior pins: genome_015 / skeleton_015 / genome_014 / acquire_014 / family_014 / kappa_013 / genome_011

## Reproduce

```bash
python -m experiments.run_tm016relate --verify-prereg
python tests/test_tm016relate.py
python -m experiments.run_tm016relate --write-lock
```

## Refuse

Pruning losers; `support>=N` earn flags; RELATE reading focus; cross-life Y→X via leftover prev; harness `setattr` to clear transients; requiring `reset_rho` between lives; universal one-observation ban; pair-hop/`observe_symbol` as the 0.16 binding channel; FAMILY 288; LOOKAHEAD; pixels; stamp 0.0.005; rewriting prior locks; putting freeze SHAs into the prereg.

## Honest scope

Not vision/hearing. Not latent object identity. Not “the organism inferred the Platonic map.” Evidence selection among inspectable candidates under ambiguity — a step beyond TM.0.15 transcription, not a sensory cortex.
