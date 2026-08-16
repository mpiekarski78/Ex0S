# CORTEX v2 D1–D2 gate contract

**Lab:** TM.0.23.CORTEX.V2.GATE  
**Product:** Ex0S **0.0.004** · `earned_next=false` · `ex0s=null`  
**Scorers:** existing honest [`experiments/cortex_develop_scorers.py`](../experiments/cortex_develop_scorers.py) `score_d0` / `score_d1` / `score_d2` — **thresholds frozen; do not soften**.

## Stages

| Stage | Pass rule (frozen) |
|-------|-------------------|
| D0 | Birth leakage gate from `score_d0` (binomial fail-to-reject + empty S). Systematic birth answer preference fails the **entire** gate. D0 is not green merely because one organism randomly HOLDs. |
| D1 | `press >= 3` and `press > harm` and `cf_differs` after teach/probe |
| D2 | `rho_ok` and `beneficial >= 3` and `holds >= 5` |
| Always-HOLD | Fails D1 and D2 |

Stop after D2. **No D3–D12.**

## Pair clearing

A pair clears **only** when:

- main clears **D1 and D2**, **and**
- twin clears **D1 and D2**

D0 must pass on both main and twin for the pair to be eligible; any D0 fail fails the pair and contributes to global gate failure if systematic.

## Gate clear

```text
sensorimotor_association_gate_clear ⇔
  (≥13/16 complete main∧twin D1∧D2 pairs clear)
  ∧ (no systematic D0 birth-leakage failure)
```

## Isolation

- Distinct eval commitment from DEVELOP.  
- Revealed gate worlds become diagnostic-only after reveal; a failed candidate cannot be edited and rescored on them.  
- Full D0–D12 requires a **later** fresh full-development commitment on new worlds.
