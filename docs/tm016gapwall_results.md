# TM.0.16.GAPWALL results

**Recorded:** frozen G0–G5 → **6/6**

- Product: `0.0.004`
- `earned_next=false`
- `ex0s=null`
- Organism: frozen ALIASFINGER-on; `agent.py` unchanged

| Cell | Result | Honest reading |
|------|--------|----------------|
| G0 adjacent | PASS | Unique adjacent route; continuity not at issue. |
| G1 empty skip | PASS | Bridge is existing empty-event skip semantics, not learned object continuity. |
| G2 episode gap | PASS | No cross-episode frontier bridge. |
| G3 distractor | PASS | The distractor is authored into the route; `a` is not privileged. |
| G4 one reappear | PASS | Measured `unique`; if unique, it is skip-driven only. |
| G5 two reappear | PASS | Equal peers tie; behavior HOLDS without choosing either. |

## Bounded fact

Frozen ALIASFINGER-on preserves the pre-gap bag across an empty event because the empty event is skipped. It loses that frontier at an episode boundary, routes through a visible distractor rather than preserving an object, and cannot resolve two equally supported post-gap candidates.

This is a capacity wall, not evidence of learned object continuity.

## Next

Opt-in persistence candidate under the frozen continuity-evidence contract ([`continuity_evidence_contract.md`](continuity_evidence_contract.md)). Alias fingerprints remain separate.

## Reproduce

```bash
python -m experiments.run_tm016gapwall --verify-prereg
python tests/test_tm016gapwall.py
```
