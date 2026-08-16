# CORTEX v6 architecture amendment

Authorized **only** by frozen [`cortex_diagnosis.v5.lock`](cortex_diagnosis.v5.lock).

## Diagnosis (do not reopen v5 worlds)

1. **No-consequence asymmetry:** motor RNG is unique per organism (32/32, no vector collisions). Birth ACT-query argmax is **not** slot-0 biased (3/32). After neutral teaching, the **first-bound** handle wins 14–0 regardless of which role is bound first. Unnormalized N(0,1) motor vectors make credit magnitudes unequal. Uniform ACT-cost credit (body_adv≈0) collapses cosines; bind-order argmax then prefers slot 0.
2. **Swap timing:** a frozen no-consequence probe after swap still prefers A (or HOLD). The old 20-probe `apply_event` window already prefers B — **in-probe physics/credit leak**. After 40 post-swap episodes, preference does **not** clearly move to B (credit/revision failure).

## Change (authorized)

1. **Exchangeable motor slots** — unit-normalized registry vectors; tie-break via `rng_motor`, never bind-order iteration.
2. **Exact credit linkage** — pending stores the **selected motor-vector snapshot**; `W_act_query` updates that snapshot only.
3. **No-consequence neutrality** — skip motor-query credit when body advantage (before ACT cost) is ~0.
4. **Evidence-driven revision** — motor-query credit only when body consequences differ; swap probes must be frozen (no new consequence) then taught.
5. Retain: `bind_actuators([opaque_handle_id,…])`, per-organism `seed_motor`, frozen `b_op[ACT]=0.85`, `OP_COST[ACT]=0.05`, handles never sensory vocab.

## Refuse

- Softening D1/D2 scorers
- Editing [`cortex_architecture_contract.md`](cortex_architecture_contract.md)
- Reusing v5 gate worlds / DEVELOP.v5
- Edit-and-rescore v5
- Opening full D0–D12 before v6 D1–D2 ≥13/16
