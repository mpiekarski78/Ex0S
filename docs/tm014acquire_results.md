# TM.0.14.ACQUIRE results — candidate freeze (not stamped)

**Ex0S under test:** 0.0.004 (unchanged)  
**Lab:** TM.0.14.ACQUIRE  
**Date:** 2026-08-16  
**earned_next:** false — no FAMILY, no Ex0S bump, no LOOKAHEAD

**Locks:** [`acquire_014.lock`](acquire_014.lock) · [`genome_014.lock`](genome_014.lock)  
**Recorded:** `python -m experiments.run_tm014acquire` → **16/16**

## Claim (candidate)

> A frozen developmental recipe can use experienced outcomes to author provenance-sensitive contextual continuations into S over an existing relational skeleton, then later use κ to select those organism-authored continuations after ρ reset, without contextual answers being planted by the apparatus.

Skeleton edges (`X→A`, `A→Y`, `X→B`, `B→Y`) remain apparatus-planted and untagged. Life authors only contextual continuations (`Y→PRESS ctx=κA`, `Y→TUNE ctx=κB` with `source=experience_ctx`).

## Headline developmental delta

```text
BIRTH S:     X→A, A→Y, X→B, B→Y   (no ctx motors)
LIFE A:      traverse X→A→Y → HOLD; teacher PRESS+success
             → organism writes Y→PRESS ctx=κA
LIFE B:      traverse X→B→Y → HOLD; teacher TUNE+success
             → organism writes Y→TUNE  ctx=κB
RESET ρ:     both rows remain
PROBE:       X→A→Y+κA → PRESS ; X→B→Y+κB → TUNE
```

Before life, those contextual answers do not exist in S. After life, they do.

## Mechanism

- One κ engine: retain compose-local `(κ, frontier)` on HOLD — no second lived-κ F.
- One-shot clear: act start, compose start, `reset_rho`, newborn, after every `observe_outcome`.
- Teacher may pass only motor + outcome. Skeleton writer runtime-refuses `ctx`.
- `run_tm011compose.py` left byte-identical; `make_acquire()` forwards `use_acquire_ctx=True`.

## Battery

| Cell | Result |
|------|--------|
| D0 birth no ctx | OK |
| D1 life A only (B HOLD) | OK |
| D2 both coexist | OK |
| D3 reset_rho | OK |
| D4 newborn reload | OK |
| D5 wipe → HOLD | OK |
| D6 swap experience_ctx rows only | OK |
| D7 exact evidence math | OK |
| D8 different lived histories → different ctx | OK |
| D9 equal evidence → HOLD | OK |
| D10 rename fid | OK |
| D11 storage order | OK |
| D12 no apparatus ctx / teacher contract | OK |
| D13 oracle score-only agree | OK |
| D14 weights stable / no shortcuts | OK |
| D15 nasty five (stale κ, non-motor, unseen fail, …) | OK |

## Does not show

Full skeleton acquisition from raw life; FAMILY / 288 worlds; Ex0S 0.0.005; LOOKAHEAD.

## Next

If this candidate survives audit: **TM.0.14.FAMILY** (still no pre-naming of a product stamp).

## Audit (post-freeze)

- Compose clears lived before early-return when store disabled; `reset_store` clears lived.
- Teacher contract: `info` keys must be exactly `{action}`; forbidden includes `destination`.
- D6/D7/D9 restore experience rows by **byte-copy** (no apparatus tag rewrite of `ctx`).
- `verify_acquire_lock` fail-closes on genome_013 / kappa_013 / genome_014 pins; write order is genome_014 then acquire_014.
