# TM.0.24.PLASTICITYMAP results

Product remains **0.0.004**. `earned_next=false`. `ex0s=null`.

| D | Question | Passed |
| --- | --- | --- |
| D0 | Complete v28 credit chain | **False** |
| D1 | Readout expressivity | **True** |
| D2 | Forced balanced ACT exposure | **False** |
| D3 | Autonomous exploration | **False** |
| D4 | Post-REST retention | **True** |
| D5 | Renamed siblings | **False** |
| D6 | Outer search | **None** |

**Primary bottleneck:** `D0_credit_chain_incomplete`

**Next change:** Do not treat Q4 as fully repaired. Complete remaining credit-chain links before reading D2 as plasticity vs exploration.

## D0 link detail

Pass: opposite advantage; zero-elig ΔW=0; wrong-tick elig fails; credited handle logit; consolidation boundary (credited slow moves, unused tensors do not).

Fail:

- correct prior eligibility: current-tick gain 0.286 vs previous-tick 0.287 (near tie after two warms)
- later probability **and** sampled behavior: after additional balanced forced exposure, trained probe 0.35 < frozen 0.45 (balanced ben/harm schedule can wash a single beneficial credit)

## D2 / D4 / D5 note (not a D2 pass)

On the FORCE world, motor scores ranked beneficial above harmful after 40 equal cycles, but probes sampled 0 ACT on either handle (HOLD). Frozen twin still sampled some ACT. D4's REST world did produce a sampled beneficial preference (13 vs 0) that survived `rest_epoch`. D5 world 0 matched that sampled preference; renamed sibling world 1 ranked scores correctly but sampled 0 ACT.

D6 not released (D1–D3 not all green). Historical WALLMAP Q3 remains the ES record.

n stays 64. LINEAGE/WALLMAP/REACH historical. QUAL/EVAL sealed. Not a capability earn.
