# TM.0.16.ALIASFINGER results

**Recorded:** canonical A0–A6 → **7/7**

- `ok`: True
- `earned_next`: false
- `ex0s`: null
- lab: `TM.0.16.ALIASFINGER`

| Cell | ok | notes |
|------|----|-------|
| A0_wall | True |  |
| A1_weak | True |  |
| A2_convergent | True | wift |
| A3_collision | True |  |
| A4_swap | True |  |
| A5_contradiction | True |  |
| A6_causality | True |  |

## Locks

- Contract: [`alias_evidence.prereg.lock`](alias_evidence.prereg.lock)
- Prereg: [`alias_finger.prereg.lock`](alias_finger.prereg.lock)
- Candidate (pre-score): [`alias_finger.candidate.lock`](alias_finger.candidate.lock)
- Freeze: [`alias_finger.lock`](alias_finger.lock)

## Reproduce

```bash
python -m experiments.run_tm016aliasfinger --verify-prereg
python tests/test_tm016aliasfinger.py
```
