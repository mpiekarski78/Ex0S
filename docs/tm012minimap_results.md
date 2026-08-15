# TM.0.12.MINIMAP results: representation distinguishability

**Ex0S under test:** **0.0.003** (not a new stamp)  
**Date:** 15 August 2026  
**Regime:** representation distinguishability (not a behavioral scorer; not a recipe jump)  
**Run:** `runs/2026-08-15_181940_tm012minimap`  
**Locks:** [`context_012.lock`](context_012.lock) (frozen CONTEXT, never rewritten) · [`minimap_012.lock`](minimap_012.lock) (apparatus)

`earned_next`: **false** — no Ex0S 0.0.004. No organism change.

## Question

For each **preregistered** contrast: if required CONTEXT outputs differ, does a candidate representation assign different states?

Same state + different required motors ⇒ **provably insufficient**.

## Computed table

| Candidate | C0 | C1 | C2 | C3 | C4 | C5 | C6 | C7 |
|-----------|----|----|----|----|----|----|----|----|
| H0 token | D | benign | collision | collision | collision | collision | collision | unobservable |
| H1 + here | D | benign | D | collision | collision | collision | D | unobservable |
| H2 + pred | D | benign | D | D | collision | D | D | unobservable |
| H3a + origin | D | benign | D | D | D | D | D | unobservable |
| H3b + path | D | benign | D | D | D | D | D | unobservable |
| H4 + incoming fid | D | benign | D | D | collision | D | D | unobservable |

D = distinguishes required outputs. Table computed from graph structure (not encoded in the scorer).

## Claim

Among the **preregistered candidate set**, `(token, origin)` is the **least-structured sufficient candidate representation** for locked contrasts **C2–C6**.

- Bare token, arrival-here, predecessor, and incoming-fact identity are insufficient on that victory set.
- Origin and full-path both distinguish C2–C6; this battery does **not** require full path history (C5 varies both origin and path).
- **C7** is `UNOBSERVABLE_FROM_PROVENANCE` — excluded from the CONTEXT victory set.
- **H4** is an incoming-fact identity **diagnostic** (not an “upper bound”); C4 shares `A→Y`, so H4 collides.
- This is **not** a proof that literal origin is globally minimal.

## Controls

- **C0:** positive control — H0 already distinguishes (different frontiers).
- **C1:** benign reuse — same Y, same motor → `COLLISION_ALLOWED`.

## Audit notes (apparatus)

Before this recorded run:

1. **C7 role was vacuous** — `score_contrast` returned `unobservable` without checking that provenance states actually collide. Fixed: differing states or matching motors ⇒ `apparatus_error`.
2. **Benign / provenance motor checks** — C1 now requires equal motors; C0 and C2–C6 require unequal motors. Vacuous same-motor provenance contrasts are `apparatus_error`.
3. **Outgoing answer fid** — `refuse_answer_derived_fid` emits `inadmissible_answer_id` (never a distinguish).
4. **CONTEXT pin** — minimap verify also requires `seed` / `seed_list_sha` equality with `context_012.lock`.

## Next (not this pass)

Same-S origin-vs-path discriminator landed as **TM.0.12.PATHDISC**; endpoint provenance falsified as **TM.0.12.MIDPATH** ([`tm012midpath_results.md`](tm012midpath_results.md)). Next after MIDPATH: route-signature minimality — not genome yet.

## Reproduce

```bash
python -m experiments.run_tm012minimap --write-lock   # once
python -m experiments.run_tm012minimap --seed 12345
python tests/test_tm012minimap.py
```
