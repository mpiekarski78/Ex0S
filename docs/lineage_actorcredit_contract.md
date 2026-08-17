# Lineage ACTORCREDIT contract — TM.0.24.ACTORCREDIT

**Lab:** TM.0.24.ACTORCREDIT  
**Subtitle:** Action-owned delayed credit  
**Product:** Ex0S **0.0.004**  
**Flags:** `earned_next=false`, `ex0s=null`, `eligible_for_000005=false`  
**Live candidate after implement:** v29 (v28 remains historical)

Not a lineage rescore. Not a capability earn. n stays **64**.

## Factorization

```text
P(beneficial action) = P(ACT) × P(beneficial handle | ACT)
```

`W_op` owns the first factor. `W_act_query` owns the second, conditional on ACT. Motor-score ranking is not equivalent to performing the beneficial ACT.

## Clamped vs passive

- **Clamped organism action:** cortex forms the pending trace on the selection tick; apparatus may replace the sampled op/handle for balanced coverage; credit still uses saved eligibility.
- **Passive imposed movement:** no pending organism action, therefore no actor credit.

Forced exposure must use clamp, never a host teaching oracle of token meaning.

## Frozen cells (fresh worlds)

| Cell | Required result |
| --- | --- |
| A0 | Zero eligibility ⇒ no fast or slow actor update |
| A1 | Organism-authored/clamped action receives credit; passive world effect does not |
| A2 | Orthogonal adjacent states; only the saved action tick receives credit |
| A3 | Repeated observation cannot apply the same credit twice |
| A4 | Beneficial and harmful consequences move policy oppositely |
| A5 | Chosen handle changes; unrelated handle does not |
| A6 | Beneficial experience increases later `P(ACT)` |
| A7 | Beneficial handle increases given ACT |
| A8 | Both probability and sampled beneficial ACT increase |
| A9 | Rename, bind-order, and consequence swap follow experience |
| A10 | Earned behavior survives intended REST/consolidation |
| A11 | Ambiguous / no-evidence situations still HOLD |

D1 remains readout expressivity, not L0 representability. D4 remains one-world retention evidence, not a general REST clear.

## After the cells

1. If A0–A11 pass, re-run developmental reachability on unused `TM024.ACTORCREDIT.REACH.FIT.` / `CHECK.` domains.
2. Only if that reachability passes, revisit Q3 and consider a new lineage commitment.
3. If reachability still fails despite complete behavioral credit, investigate state/developmental dynamics — not n.

## Refuse

L0-specific circuitry; QUAL/EVAL reveal; rewriting LINEAGE/WALLMAP/REACH/PLASTICITYMAP/v28 locks; `earned_next`; 0.0.005; moving `τ`/`δ`; increasing n; FULLDEV.R7; treating clamp as a teaching oracle; another lineage run before A-cells and reachability.
