# TM.0.13.CONTEXT results: first cognitive function in M

**Ex0S under test:** **0.0.003** (not a new stamp)

**Recorded run:** `runs/2026-08-15_221556_tm013context`

**Battery:** 14/14 cells OK

`earned_next`: **false** — no Ex0S 0.0.004. `genome_011.lock` immutable.

## Claim (candidate, not stamped)

> The organism carries compact transient context while traversing externally acquired knowledge, allowing identical frontier symbols to support different learned continuations without encoding those continuations in the genome.

This pass **installs** provenance-sensitive composition under `use_context_kappa=True`. Product stamp stays **Ex0S 0.0.003**. A later **TM.0.13.FAMILY** (generated worlds) is required before discussing 0.0.004.

## What was frozen

| Artifact | Role |
|----------|------|
| [`kappa_013.lock`](kappa_013.lock) | Scientific κ contract + `ctx_encoding=ksem-sha256-v1` + known-answer vectors |
| [`genome_013.lock`](genome_013.lock) | CONTEXT-on candidate (`use_context_kappa=true`, `kappa.py`, current agent) |
| [`context_013.lock`](context_013.lock) | Family apparatus pins + refuse list |
| [`genome_011.lock`](genome_011.lock) | **Unchanged** — historical 0.0.003 freeze at `c392aa…` |

## Mechanism

- κ initialized and advanced **only** after a selected **non-motor** hop.
- Hop-1 motor: return motor; **no** κ.
- Motor after κ exists: match `(Y, κ)`; do **not** step κ.
- On a derived frontier: filter **unvisited** facts first; if **any** remaining eligible has `ctx`, discard untagged; exact `ctx==κ` only; zero matches → **HOLD**.
- Planted family `ctx` comes from independent `reference_route_kappa` (not sole live `kappa.py`).
- Route order is **evidence-causal** (visited fact_ids), not apparatus dual-traces.

## Audit fixes (this pass)

- Visited `ctx` facts no longer poison later untagged matches at the same bind.
- `C13_donor_revise` revises evidence **in place** on one S (not a duplicate of route-order).
- `verify_context_lock` fail-closes on `cell_ids` / `n_ok` / all reference helper SHAs / `genome_013` pin.
- Feature-off cell proves untagged high-support wins when κ filtering is off.

## Family table

| Cell | Result | Note |
|------|--------|------|
| C13_route_order | OK | A-then-B → PRESS; B-then-A → TUNE |
| C13_c7_tie_hold | OK | Equal-evidence same-κ motors → HOLD |
| C13_wipe | OK | Empty S → HOLD |
| C13_donor_revise | OK | In-place support swap PRESS → TUNE |
| C13_retarget | OK | Swap ctx tags → output changes |
| C13_reset_rho | OK | Same contextual motor after ρ reset |
| C13_fid_rename | OK | Semantic κ unchanged under fid rename |
| C13_ctx_beats_untagged | OK | ctx support=1 beats untagged support=1000 |
| C13_no_fallback_untagged | OK | Mismatch + untagged → HOLD (κ asserted) |
| C13_hop1_motor_no_kappa | OK | Direct motor; κ stays None |
| C13_visited_ctx_no_poison | OK | Consumed ctx does not block untagged revisit |
| C13_depth_holdout | OK | Preregistered DEPTH-shaped vector C |
| C13_new_nonce_order | OK | Fresh tokens; same topology split |
| C13_feature_off_untagged_wins | OK | ON→PRESS; OFF→untagged TUNE |

## Compatibility

HEAD with `use_context_kappa=False` keeps **0.0.003 behavioral compatibility** recipe (compose-on). Historical freeze verifies recipe blobs at baseline commit `c392aa515b7a3445bb15bc55ad969d971632ea3f` against immutable `genome_011.lock`. HEAD `agent_sha` **differs** from that lock — reported honestly; not rewritten into 0.0.003.

## Not claimed

- κ solves here-split (C2) or evidence traps (C6)
- Authoring `ctx` from a free life
- Ex0S 0.0.004
- 252-world CONTEXT generator (that is **TM.0.13.FAMILY**)

## Reproduce

```bash
python -m experiments.run_tm013context
python tests/test_tm013context.py
python tests/test_tm011family.py
```
