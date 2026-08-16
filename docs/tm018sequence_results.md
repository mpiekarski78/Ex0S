# TM.0.18.SEQUENCE results

**Recorded:** expressive life → **PASS**

- Product: `0.0.004`
- `earned_next=false`
- `ex0s=null`
- Mechanism: `use_symbol_sequence` / `experience_sequence`

## Capacity

Cleared through **E12** (main + twin + capacity). Life probes last cleared **E11**; E12 is the capacity-launch stage.

| Lane | ok | last_stage_clear | probes |
|------|----|------------------|--------|
| main | True | E11 | 13 |
| twin | True | E10 | 5 |

## Unconfounded capacity lanes

| Lane | ok | first_fail_rung |
|------|----|-----------------|
| vocab_lane | True | None |
| length_lane | True | None |
| age_lane | True | None |

## Wall metrics

- main: `{"s_row_count": 176, "n_emit_timings": 12, "p50_emit_s": 7.521640509366989e-05, "p95_emit_s": 0.00012054387480020523, "complete_utterance_latency_s": 0.00024323631078004837, "evidence_rows_examined_p50": 84.0}`
- twin: `{"s_row_count": 132, "n_emit_timings": 5, "p50_emit_s": 4.3472275137901306e-05, "p95_emit_s": 0.0013943230733275414, "complete_utterance_latency_s": 0.0005702221766114235, "evidence_rows_examined_p50": 50.0}`

## Bounded fact

A frozen, generic mechanism learned termination and variable-length ordered construction from factorized grounded evidence, composed an unseen utterance, followed a counterfactually reordered language, and remained causally dependent on grounding and sequence evidence in S.

## Next

Dialogue wall is diagnostic only. No Ex0S 1.0 / product stamp.

## Reproduce

```bash
python -m experiments.run_tm018sequence --verify-prereg
python tests/test_tm018sequence.py
```
