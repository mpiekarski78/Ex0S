# TM.0.36.UPDATEGEOM contract

**Lab:** TM.0.36.UPDATEGEOM · **Not v39.** No neural edit. v38 closed. TM035 one-shot-only v39 stayed closed.

Fresh `TM036.UPDATEGEOM.DEV./TWIN.` worlds. Do **not** edit TM031–TM035 locks. Do **not** change rehearsal cap, write L2, `R`, or `ACT_RECALL_MODES`.

## Why this wall

TM035: write is safe; one-shot plasticity damages established rows; rehearsal plasticity independently damages established rows; TM032 post-snapshot replay can still recover. That is a stability–plasticity conflict inside shared `_apply_act_query_update`, not a one-shot-only bug and not a replay-budget/timing bug.

## Parent

Teach credits **0–6** with default v37. Apply TM035 `write_only` for credit 7. Checkpoint. Every geometry arm clones from that identical post-write snapshot. Protected rows = non-violating stored rows on that snapshot (the new slot is typically still a violator and is not protected).

## Interference matrix

From the post-write snapshot, for each stored row \(i\), clone and apply **one** native `_apply_act_query_update` targeting \(i\).

\[
I_{ij}=\Delta\mathrm{margin\ of\ row\ }j
\]

Record pre/post geometric margins. Do not treat \(I\) as a halfspace.

## Intervention arms (runner-side, same snapshot)

- `native` — ranking-error one-shot, then unchanged v37 sequential (Gauss–Seidel) burst.
- `jacobi` — each pass: compute every currently violating-row update against the **same frozen** \(W\), sum, apply once; up to 16 passes; stop at zero violations.
- `protect` — deterministic dyadic safe scaling (\(\alpha=1,1/2,1/4,\ldots\) until no representable weight change) on **every** ACT-query update (one-shot and rehearsal). Largest \(\alpha\) that keeps every protected row non-violating **and** strictly improves the targeted row’s geometric margin; else \(\alpha=0\) and skip.
- `oracle` — closest \(W\) (Frobenius) satisfying all stored-row ranking+margin constraints. Diagnostic only. Not an organism law.

Fix gate: store violations 0, live ranking 8/8. Margin is recorded, not a live gate.

## Routes (first-match on diagnostic worlds)

Diagnostic = `native` fails the fix gate.

1. `updategeom_no_native_fail` — every world already fixes under native
2. `updategeom_jacobi_sufficient` — Jacobi fixes every diagnostic world (sequential order/path dependence)
3. `updategeom_protect_sufficient` — protect-every-update fixes every diagnostic world; Jacobi does not
4. `updategeom_oracle_only` — only the closest feasible \(W\) fixes; local update geometry is insufficient
5. `updategeom_capacity_wall` — oracle itself fails stored-row margins
6. `updategeom_mixed_routes` — diagnostic worlds disagree

**v39 is not frozen here.** The decision only names which law would be the next freeze.

Product **0.0.004**.
