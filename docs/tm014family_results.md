# TM.0.14.FAMILY results — kill-or-earn on frozen ACQUIRE

**Ex0S under test:** 0.0.004 (product stamp unchanged)  
**Lab:** TM.0.14.FAMILY  
**Date:** 2026-08-16  
**Recorded:** `runs/2026-08-15_232226_tm014family` → **288/288**  
**Earn:** `earned_next=true`, **`ex0s=null`** (no product version named by the runner)

**Lock:** [`family_014.lock`](family_014.lock) (pins `genome_014` / `acquire_014` / `kappa_013` + sealed E–H world manifests + per-row `holdout_row_shas`)

## Audit hardenings (this pass)

- `route_association` refuses invented lives (sealed `counterfactual_life` only)
- Counterfactual measure requires **behavioral** divergence (not list inequality alone)
- `newborn_between` no longer stamps True without work
- Oracle: `press↔κa`, `tune↔κb`
- Diamond routes: hop count equals declared `depth`
- Earn gate pins `family_014.lock`, `DEFAULT_SEED`, and lock-committed holdout row SHAs
- Docs: product-stamp wording = FAMILY earn (not ACQUIRE earn)

## Regime

**Generated developmental histories** — apparatus supplies birth skeleton, route/evidence conditions, teacher motor, and outcome. Organism produces κ, authors `source=experience_ctx` continuations, and uses them later. Not open-ended exploration; not apparatus-planted `ctx`.

## Preregistered claim

> A frozen developmental recipe generalizes across unseen generated life histories by converting experienced outcomes into provenance-sensitive contextual continuations in S, which persist across working-state reset and causally steer later behavior without contextual answers being supplied by the apparatus.

## Headline

| | |
|--|--|
| All worlds | **288/288** |
| Developed A–D | **144/144** |
| Hold-out E–H | **144/144** |
| Birth ctx rows | **0** (all worlds) |
| Unexpected ctx writes | **0** |
| Genome / cortex | stable (`genome_014`) |
| Product stamp | **none** (`ex0s=null`) |

## Per-family

| Family | Role | Solved |
|--------|------|--------|
| A | develop — one/two contextual experiences | 36/36 |
| B | develop — repeated success / evidence | 36/36 |
| C | develop — contradiction/revision | 36/36 |
| D | develop — clutter + irrelevant outcomes | 36/36 |
| E | hold-out — deeper lives | 36/36 |
| F | hold-out — multiple κ at frontier | 36/36 |
| G | hold-out — interleave + newborn | 36/36 |
| H | hold-out — order-equiv + route association | 36/36 |

## Global invariants

Every world: `birth_no_ctx`, `authored_after_life`, `persist_rho`, `persist_newborn`, `strip_experience_hold`, provenance ledger (`ctx` ↔ `experience_ctx`), `weights_stable`, `genome_delta`, `no_shortcut_writes`.

Branch measures (`donor_transfer`, `counterfactual_life`) required across every family (coverage floor), not necessarily every individual world.

## Holdout discipline

E–H world manifests (full World: birth + primary/counterfactual/donor lives + probes + interventions) committed in `family_014.lock` **before** organism answers. CI: A–D smoke + E–H sealed checks only — no `make_acquire` / traverse / teacher on E–H pre-canonical.

## Shows / does not show

**Shows:** same frozen ACQUIRE recipe creates different durable contextual knowledge across unseen generated lives; behavior follows organism-authored rows after ρ/newborn; strip removes learning; donor/counterfactual causal branches; order-equivalent evidence vs route-association swap.

**Does not show:** open-ended exploration; skeleton acquisition from life; LOOKAHEAD; a named Ex0S product bump.

## Next

Acquire the **relational skeleton** itself via observed-transition acquisition (TM.0.15.SKELETON — done). Next interesting attack: unstructured/ambiguous events. Product stamp naming remains a human decision.

## Reproduce

```bash
python -m experiments.run_tm014family --verify-sealed
python tests/test_tm014family.py
python -m experiments.run_tm014family --canonical --workers 8
```
