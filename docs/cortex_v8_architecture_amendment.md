# CORTEX v8 architecture amendment

Authorized **only** by [`cortex_diagnosis.v7.lock`](cortex_diagnosis.v7.lock) + [`cortex_diagnosis.v7.note.lock`](cortex_diagnosis.v7.note.lock) and [`cortex_v7_gate.audit.lock`](cortex_v7_gate.audit.lock).

## Diagnosis (do not reopen v7 gate worlds for edit-rescore)

1. C4 revision is real.
2. Population C5/C6 already show consequence-dependent initial learning.
3. 0/16 is a per-life-extra fail, not a missing-learning fail.
4. Sampled D1 press=0 lives still ACT on frozen probes after 30 teach.
5. No new neural credit law is authorized.

## Change (authorized)

1. **Retain v7 neural** (`skip_act_cost` when `body_adv≈0`; handle-keyed unit vectors; zero `W_act_query` at birth; snapshot credit).
2. **Scorer grain:** birth-weight paired extras on frozen probes; `life_delta_min=0.10`; D1 floors on that same probe.
3. **Worlds:** derive 16 gate pairs from the revealed eval seed.
4. **Reveal:** pin the pushed git commit SHA of the candidate.

## Refuse

- Softening floors
- Editing `docs/cortex_architecture_contract.md`
- Edit-and-rescore v7
- DEVELOP before v8 gate ≥13/16
- Opening full D0–D12 before that gate
- Neural retune to force every frozen life below deterministic D1
