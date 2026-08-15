# TM.0.12.ROUTESIG.IDENTITY results: storage-row fid vs relation identity

**Ex0S under test:** **0.0.003** (not a new stamp)  
**Date:** 15 August 2026  
**Regime:** representation-contract (IDENTITY on frozen C10)  
**Run:** `runs/2026-08-15_212750_tm012routesig_identity`  
**Locks:** [`routesig_identity_012.lock`](routesig_identity_012.lock) · pins CONTEXT + MINIMAP + PATHDISC + MIDPATH + [`routesig_012.lock`](routesig_012.lock) + [`routesig_depth_012.lock`](routesig_depth_012.lock) (never rewritten)

`earned_next`: **false** — no Ex0S 0.0.004. No organism / genome / CONTEXT-in-M this pass. No alpha-rename.

## Question

Does context need **persistent fact identity** (database row / fid), or **semantic relation identity** (what was traversed)?

Same rolling F as ORDER (`kappa_seed` / `kappa_step`). The payload is the question.

## Perturbations

**C12A — opaque fid rename.** Clone frozen C10; bijectively rename only fids. Hold fixed: bind, did, role, here, support, route order, motors.

**C12B — same fid, changed relation.** Keep path fids; rewrite the last hop `P→Y` to `P→Z` under the same fid. Prefix tokens unchanged. Not a token-wide alpha-rename.

## Accumulators

| ID | Payload |
|----|---------|
| Kfid | origin + ordered path **fids** |
| Ksem | origin + ordered **canonical(bind, did)** |

Ksem excludes fid, here, support, role, motor, answer. Evidence and locality stay in S / selector machinery.

## Computed table

| Candidate | C12A fid-rename | C12B same-fid rewrite |
|-----------|-----------------|------------------------|
| Kfid (storage row) | differs | same |
| Ksem canonical(bind, did) | same | differs |

ORDER still holds on both encodings: Kfid and Ksem distinguish C10 route_a vs route_b.

C12B is a diagnostic if Ex0S revision always mints a new fid — it still forces the metaphysics into the lock.

## Claim

**Fid-based κ is storage-identity dependent. Semantic κ tracks directed relation identity `canonical(bind, did)`.** Same F; different payload. Kfid changing under export/import of the same knowledge is not a cognition failure — it is bookkeeping dependence.

**Not:** Kfid fails cognition; κ must be invariant under token alpha-rename; SHA-as-genome; 0.0.004.

## Audit notes (apparatus)

After the recorded scientific table:

1. **One-trace geometry** — C12A/C12B only checked route_a. Fixed: both C10 traces.
2. **C12B inherited identity soft** — path-fid equality did not prove exactly one `(bind, did)` rewrite with fid kept. Fixed: inherit every left fid; exactly one pair changes; Z→motors are the only adds.
3. **ORDER F not cross-pinned** — IDENTITY could drift from ORDER’s `kappa_*` / `edge_fid` while matching its own lock. Fixed: live SHA must equal `routesig_012.lock`.
4. **Shuffle / unique fids** — file-order independence and unique fids claimed but not enforced. Fixed.
5. **Z→motor insertion order** — set iteration. Fixed: sorted motors.

Scientific table unchanged after fixes.

## Out of scope (deliberate)

Token renaming (`DOG→MAMMAL` to `X17→X42`) is graph isomorphism / equivariance, not this lab. Do not require identical digest bits after alpha-rename.

## Next (not this pass)

Freeze the κ contract (payload = relation identity, F already pinned). Then the first CONTEXT function in M: carry compact route state while traversing acquired knowledge. Attack it with generated worlds before any 0.0.004 stamp. ALPHA/structural rename remains optional and later.

## Reproduce

```bash
python -m experiments.run_tm012routesig_identity --write-lock   # once
python -m experiments.run_tm012routesig_identity --seed 12345
python tests/test_tm012routesig_identity.py
```
