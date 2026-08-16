# CORTEX v7 architecture amendment

Authorized **only** by frozen [`cortex_diagnosis.v6.lock`](cortex_diagnosis.v6.lock) + [`cortex_diagnosis.v6.note.lock`](cortex_diagnosis.v6.note.lock) and [`cortex_v7_stat_contract.lock`](cortex_v7_stat_contract.lock).

## Diagnosis (do not reopen v6 gate worlds for edit-rescore)

1. **C4 revision is real.** Frozen immediate probe stays A; 40 teach → B; restore A.
2. **Population C5 already shows consequence-dependent learning** (29/32 trained > frozen, mean Δ≈0.23). The single-life D1 bar (press=11, 6>4) is not a learning test.
3. **Population C6 is already neutral** (slot and post-hoc label effects inside the preregistered band).
4. **Gate 7/16** fails from D1 always-HOLD (`press=0,harm=0`) and D2 `holds<5`, not from an 8–0 bind-order win.
5. No-consequence `OP_COST[ACT]` still updates `W_op` (`adv = −0.05`), which can extinguish ACT before body evidence arrives.

## Change (authorized)

1. **Retain v6 motor geometry:** handle-keyed unit vectors; zero `W_act_query` at birth; snapshot credit; skip motor-query credit when `body_adv≈0`; bind-order-independent tie-break.
2. **Skip `W_op` ACT credit when `body_adv≈0`** — no-consequence cost must not punish ACT. Prediction credit unchanged. Nonzero body advantage still three-factor updates `W_op` and `W_act_query`.
3. **Boundary / gate use the statistical contract** — paired plasticity, permutation / CI, trained > birth and trained > frozen, D2 association. Do not require every frozen individual to fail `press≥3 ∧ press>harm`.
4. Retain `bind_actuators`, `seed_motor`, `b_op[ACT]=0.85`, `OP_COST[ACT]=0.05`, empty `MOTOR_ACT_TOKENS`.

## Refuse

- Softening D1/D2 floors
- Editing `docs/cortex_architecture_contract.md`
- Edit-and-rescore v6
- DEVELOP.v6 / DEVELOP.v7 before v7 gate ≥13/16
- Opening full D0–D12 before that gate
