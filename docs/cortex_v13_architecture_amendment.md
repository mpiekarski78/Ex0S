# CORTEX v13 architecture amendment

Authorized only by [`cortex_diagnosis.v12.lock`](cortex_diagnosis.v12.lock).

One generic slow advantage baseline: compare credited ACT `body_adv` to a decaying baseline of recent agreeing advantages. Opposite-sign mismatch raises the existing HOLD response and does not snap the baseline. Retain v12 scorers, floors, bias, and swapped D2 window. Do not lower `holds>=5`. Do not raise `CONFLICT_HOLD_BIAS`. No capability-specific functions.
