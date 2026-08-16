# TM.0.23.CORTEX v5 diagnosis

Observational on frozen candidate v5. Does not rescore revealed gate worlds.

## No-consequence asymmetry

- motor seeds unique: `True` (32 seeds / 32 organisms)
- vector collisions: `0`
- slot-0 birth ACT-query argmax: `3/32` (0.094)
- motor vector ‖v‖ mean/std: `5.677` / `0.737`

## Swap timing

- learned A: `True` counts `{'h_d6d045e64b97': 14}`
- immediate frozen first token: `None` (is A: `False`)
- immediate frozen 8-probe: `{'h_d6d045e64b97': 4}`
- contaminated 20-probe (old C4): `{'h_d6d045e64b97': 3, 'h_0212ab081a3e': 4}`
- later after 40 swap episodes: `{'h_d6d045e64b97': 9, 'h_0212ab081a3e': 8}` pref B `False`
- leak into first selection: `False`
- stale window contaminated: `True`
- credit fails to revise: `True`

## Ranked causes

1. **bind_order_neutral_preference** — After no-consequence teaching, the first-bound handle is preferred — slot/order bias, not physics.
2. **unnormalized_motor_vectors** — Motor-registry vectors are N(0,1) not unit-normalized (cosine is scale-invariant; norms still vary).
3. **stale_probe_applies_new_consequences** — Old C4 immediate 20-probe used apply_event under swapped physics, so credit+body leaked into the 'stale' window.
4. **credit_fails_post_swap_revision** — Immediate frozen probe stays A but 40 post-swap episodes do not move preference to B.

V6 authorized only by this lock. Full D0–D12 stays closed until a fresh v6 D1–D2 gate ≥13/16.
