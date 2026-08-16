# TM.0.19.INQUIRE results

**Recorded:** inquire life → **PASS**

- Product: `0.0.004`
- `earned_next=false`
- `ex0s=null`
- Mechanism: `use_inquire` / `plan_inquiry` / `experience_inquire`

## Capacity

Cleared through **I12**. Life probes last cleared **I12**; I12 includes capacity launch.

| Lane | ok | last_stage_clear | probes |
|------|----|------------------|--------|
| main | True | I12 | 15 |
| twin | True | I11 | 2 |

## Unconfounded capacity lanes

| Lane | ok | first_fail_rung |
|------|----|-----------------|
| hypotheses_lane | True | None |
| depth_lane | True | None |
| age_lane | True | None |
| source_count_lane | True | None |

## Bounded fact

An opt-in recipe may derive competing hypotheses from factorized S evidence, score one-step epistemic partition value then locked cost, and return plan_inquiry → ANSWER | PROBE_ATOMS | SYMBOLIC_ACTION | HOLD without calling the teacher. Host-executed consequences enter ordinary grounding channels; experience_inquire stores plans/traces only. Budget 8; scored depth ≤ 4; inquiry metadata alone never substitutes for world evidence.

## Audit repair

Expression via frozen `emit_sequence`; dual-memory strips consequence rows only; I12 donor fork; `inquire.candidate.v1.lock` preserved across agent rewrite.

## Next

Final wall is diagnostic (reliability / planning / goals). No Ex0S 1.0.

## Reproduce

```bash
python -m experiments.run_tm019inquire --verify-prereg
python tests/test_tm019inquire.py
```
