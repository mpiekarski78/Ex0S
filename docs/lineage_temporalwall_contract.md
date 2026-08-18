# TM.0.34.TEMPORALWALL contract

**Lab:** TM.0.34.TEMPORALWALL · **Not v39.** v38 closed unchanged.

Fresh `TM034.TEMPORALWALL.DEV./TWIN.` worlds. Do **not** edit TM032/TM033 locks. Do **not** change the v38 controller, plateau, 16-pass cap, or 44.

## Why this wall

Locked TM033 telemetry shows update quantity is not the lever: adaptive 13 updates failed; fixed extra 130 updates failed; TM032 post-snapshot 44 updates succeeded. v37 and fixed-extra diagnostic endpoints match TM032 `none`. REST on those snapshots reported store-convergence at 16/44. The lock does **not** record per-credit zeros, per-pass cross-row breaks, per-row margin deltas, first-incorrect times, or live ranking after exact TM032 `awake_only`.

## What is measured

Runner-side probes around `_gated_rehearsal_pass` (no neural edit). After each credit: violation count and slots. After each gated pass: slots that newly violate, slots that newly clear, and violating rows whose geometric margin increased without a count drop.

Exact frozen TM032 procedure on a v37 post-awake clone: `full_rest` sibling supplies `n_replay`; `awake_only` is `gated_rehearsal` imported from the frozen TM032 runner, compute-matched to that cap.

## Routes (first-match)

1. No diagnostic (every v37 post-awake already fixes) → wall uninformative.
2. Diagnostic, exact TM032 `awake_only` does not fix → domain/procedure mismatch; resolve before architecture.
3. Diagnostic, rescue holds, later credits rebreak a zero-violation store → temporal interference/protection.
4. Rescue holds or not recorded as mismatch: a pass updates one row and another previously-correct row becomes violating → conflicting rehearsal gradients.
5. Margin of a targeted row improves while `n_violations` does not drop, and 3–4 do not fire → plateau metric too coarse (new controller prereg later, **not** a v38 repair).
6. Mixed seeds.

Product **0.0.004**.
