# CORTEX v5 architecture amendment

After M_act boundary reds on candidate v4 (planted `press`/`harm` dictionary). V4 D1–D2 gate credit **does not transfer**.

## Change (authorized)

1. **Remove** birth `MOTOR_ACT_TOKENS = ("press", "harm")` lexicon planting into `motor_vocab` / sensory `vocab`.
2. **Universal actuator surface:** `bind_actuators([opaque_handle_id, ...])` — strings only.
3. Cortex **motor-registry RNG** assigns nonsemantic vectors internally:
   - independent of spelling, physics, and scorer truth
   - main/twin use **separate** `seed_motor` / `rng_motor` streams
   - motor RNG state + registry **checkpointed**
   - rebinding the **same** handle **restores** its vector
4. Handle string = external registry key only; **never** sensory/vocab neural input.
5. Physics knows handle effects; cortex does not.
6. ACT selects among bound motor-registry vectors; emitted token is the bound handle id for physics only.
7. Retain defensible innate tendencies: frozen `b_op[ACT]=0.85`, generic `OP_COST[ACT]=0.05`.

## Forbidden API

```text
bind_actuators([{"id": ..., "vector": ...}])   # runner must not supply vectors
```

## Evidence

[`cortex_mact_boundary.lock`](cortex_mact_boundary.lock) — C1–C3/C7 reds on planted dictionary.

## Refuse

- Softening D1/D2 scorers / thresholds
- Editing [`cortex_architecture_contract.md`](cortex_architecture_contract.md)
- Reusing v4 gate worlds / commitment
- Runner-supplied motor vectors
- Transferring v4 gate clear to v5
- Overwriting historical `.lock` paths
