# TM.0.23.CORTEX results: developmental artificial cortex (birth pass)

**Ex0S under test:** **0.0.004** (not a new stamp)
**Lab:** TM.0.23.CORTEX
**ok (sanity):** `True`
**learning_law_ok:** `True`

Locks: [`cortex.prereg.lock`](cortex.prereg.lock) · [`cortex_wall.prereg.lock`](cortex_wall.prereg.lock) · [`cortex_birth.lock`](cortex_birth.lock) · [`cortex.candidate.lock`](cortex.candidate.lock)

`earned_next`: **false** — no Ex0S 0.0.005. Product stamp remains **0.0.004**.

## This pass

Architecture contract, worlds, preregs, CPU/GPU birth substrate, and unscored learning-law sanity. **No D0–D12 scoring.**

## Environment pin (birth)

- `torch`: `2.13.0+cu130`
- `torch.version.cuda`: `13.0`
- device: `NVIDIA GB10`
- cuDNN: `92000`
- python: `3.12.3`

## Sanity

- `order_ab_ba`: **pass** — `[A,B]` vs `[B,A]` different sensory `ρ` trajectories
- `prediction`: **pass** — repeated transitions reduce `||ε||` under directed `W_pred`
- `advantage_path`: **pass** — beneficial body transition raises responsible-action logit path; harmful lowers
- `exploration`: **pass** — `rng_action` visits alternative ops under uncertainty
- `write_retrieve`: **pass** — WRITE then RETRIEVE changes later trajectory vs no-write
- `checkpoint`: **pass** — CPU continuation reproduces actions / RNG draws
- `rho_reset`: **pass** — slow weights kept; fast activity + retrieval buffer cleared
- `scorer_isolation`: **pass** — `scorer_only` never reaches cortex / memory / body / action updates
- `cpu_gpu`: **pass** — same discrete ops on frozen short trajectory (tolerances documented in birth lock)

## Human / math audit (birth)

Checked against [`cortex_architecture_contract.md`](cortex_architecture_contract.md):

1. Credit order: previous-action update from `eligibility_(t-1)` completes before current sensory microticks.
2. Sensory path is sequential (START + source → symbols → END → STATE sum); no message mean-pool.
3. `body_state` is four physics floats; setpoint `[1,0,1,0]`; advantage uses `||body−body*||` delta minus op `cost`.
4. Motor loop `T_max=8`; ops `{RETRIEVE,WRITE,EMIT,ACT,STOP,HOLD}`; RETRIEVE fills next-tick buffer; WRITE visible after commit.
5. Four RNG streams checkpointed; main/twin independent registry+source seeds.
6. Factory isolation: `make_cortex` does not import `make_interpret` / `ThreeMemoryAgent`.
7. Held-out secrets stay sealed locally; only `eval_seed_commitment` is in prereg; smoke does not open sealed seed.

Audit conclusion: learning-law sanity is sufficient for candidate freeze. **Not** a D-life earn; **not** a product stamp.

## Next

D0–D12 developmental scoring on a later pass (architecture-lane separate births; mature forks from n=64 only).
