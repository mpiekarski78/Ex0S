# TM.0.13.FAMILY results: Ex0S 0.0.004 — Contextual Composition

**Ex0S:** **0.0.004 — Contextual Composition**

**Date:** 2026-08-15 (UTC recorded run)

**Regime:** planted / generated contextual S (not experience-acquired; that is TM.0.14 ACQUIRE)

**Recorded run:** `runs/2026-08-15_223308_tm013family` (earn); reconfirm after audit: `runs/2026-08-15_223858_tm013family`

**Lock:** [`family_013.lock`](family_013.lock) (pins `genome_013` / `kappa_013` / `context_013` / `genome_011` + holdout world manifests)

## Claim

> A frozen CONTEXT recipe carries bounded provenance-sensitive state through externally acquired relation graphs and uses that state to distinguish otherwise identical frontiers across unseen generated world families, while acquired continuations remain in S and cognitive weights remain unchanged.

## Headline

| | |
|--|--|
| All worlds | **288/288** |
| Developed A–D | **144/144** |
| Hold-out E–H | **144/144** |
| Genome changes during run | **0** (`genome_013` start+end) |
| Apparatus interventions on organism | **0** |
| Holdout regenerations before earn | **1** (see below) |

## Per-family

| Family | Role | Depth | Solved |
|--------|------|-------|--------|
| A | develop — route-order | 6 | 36/36 |
| B | develop — shared-mid converge | 4 | 36/36 |
| C | develop — evidence + untagged trap | 6 | 36/36 |
| D | develop — clutter + storage order | 6 | 36/36 |
| E | hold-out — greater depth | 7–8 | 36/36 |
| F | hold-out — cycle revisit | 8 | 36/36 |
| G | hold-out — nonce vocabulary | 7–8 | 36/36 |
| H | hold-out — mixed adversarial | 7–8 | 36/36 |

## Causal interventions (every world)

All mandatory measures present and green: `context_route`, `ctx_beats_untagged`, `ctx_no_fallback`, `tie_hold`, `retarget_ctx`, `revise_evidence`, `revise_route`, `s_necessity`, `rho_reset_same_agent`, `newborn_reload`, `storage_identity_order_invariance`, `feature_off_compat` (exact 0.0.003 expectation), `no_shortcut_writes`, `weights_stable`, `genome_delta`.

Formal causal headline (planted-S wording):

1. same genome + same cue + different **planted S history** → different κ → different continuation  
2. same S except ctx retargeted → behavior follows retarget  
3. wipe → HOLD; same-agent `reset_rho` and newborn reload both preserve contextual motor; feature off → exact bind/evidence outcome; storage-identity order change → semantic invariance  

## Holdout discipline

E–H world manifests were cryptographically committed in `family_013.lock` **before** organism answers. CI only hash/schema/oracle-checks E–H.

**Apparatus note (not a genome change):** the first canonical attempt (`runs/2026-08-15_223141_tm013family`) failed F (0/36) and part of H because the cycle generator left equal-evidence ties at `q`. Holdout generators F/H were fixed (evidence ladder `qa/qb > qz > qy`), **fresh holdout manifests** were written (prior peek invalidated), and the recorded earn run is `223308` — first behavioral contact with the committed worlds that earned the stamp.

## Audit fixes (post-earn)

- Refuse behavioral E–H contact outside full canonical dims (dead `pass` removed).
- Earn gate checks each scored holdout `manifest_sha` against the committed row-set.
- A/B/C structurally distinct (B: shared-mid converge; C: planted untagged → feature-off `flip`).
- `no_shortcut_writes` also covers wipe / ρ-reset probes; newborn asserts κ.
- Sealed verify uses explicit returns (not `assert`).

Organism / E–H world generators unchanged after the earn-run commitment; A–D + verification apparatus updated and lock re-pinned.


**Shows:** frozen CONTEXT-on recipe generalizes across 288 unseen planted contextual worlds with interventions; acquired continuations stay in S; cortex weights unchanged.

**Does not show:** experience authoring contextual S (TM.0.14 ACQUIRE); lookahead/backtracking; “abstraction” or “reasoning.”

## Reproduce

```bash
python -m experiments.run_tm013family --verify-sealed
python tests/test_tm013family.py   # A–D smoke + E–H sealed only
python -m experiments.run_tm013family --canonical --workers 8
```
