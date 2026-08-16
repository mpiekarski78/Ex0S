# TM.0.16.PERSIST results

**Recorded:** canonical C0–C6 → **9/9**

- Product: `0.0.004`
- `earned_next`: false
- `ex0s`: null
- lab: `TM.0.16.PERSIST`

| Cell | ok | notes |
|------|----|-------|
| C0_gapwall | True | GAPWALL G1/G2/G5 reused; skip is not continuity |
| C1_weak | True | reappear / apply-only / read-only HOLD; no permission |
| C2_mark | True | unique apply→read earns identity-dependent cue norb → wift |
| C3a_both | True | both verify; refuse unique |
| C3b_neither | True | neither verifies; refuse unique |
| C3c_conflict | True | norb on then off; refuse unique; rows retained |
| C4_swap | True | mk_beta swaps the verifying candidate |
| C5_contradiction | True | later state / second read / second apply withdraw; reprobe HOLD |
| C6_causality | True | reset ρ retains; strip/donate follow S only |

## Bounded fact

Opt-in continuity rows in S permit a one-hop use-time projection only when exactly one apply and exactly one matching read remain unique. Contradiction withdraws permission on recompute without deleting rows. Product stays 0.0.004.

Audit repair (v1 candidate superseded): recompute enforces the phase–operation lock; a multi-token cue that includes a permitted Q HOLDs instead of falling through to raw compose; HOLD means no dest chosen; cells emit the frozen prereg fixtures; fingerprint rows cannot substitute for continuity evidence.

## Next

No product stamp. Alias fingerprints remain a separate track. LIFEWALL continuous-lifetime wall: [`tm016lifewall_results.md`](tm016lifewall_results.md). Anonymous features / encoders later.

## Reproduce

```bash
python -m experiments.run_tm016persist --verify-prereg
python tests/test_tm016persist.py
```
