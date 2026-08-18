# TM.0.44.MEMPROJ results

## Decision: `memory_not_necessary`

Product **0.0.004**. No `cortex.candidate.v41.lock`. Novelty/familiarity is not in the claim.

First-match on the frozen ladder: the symbolic-oracle ceiling held (2/2 worlds), then the `no_persistent_memory` arm solved the same 2-cue/2-handle association. Persistent opaque S was therefore not necessary for the tested behavior.

DEV ran once on clean `580588f`. Frozen runner SHA `9bbde3eafd7c56ea2a39835405fe78221687a49c01ace0261a41710db7a2cfd0`. 17 cells. SOCP stayed off. `joint_socp.py` was not edited.

## Cell brief

- Oracle associate: pass on w0 and w1.
- `no_persistent_memory` associate: pass on w0 and w1.
- Learned and birth associate: 1/2 cues on both worlds; fail.
- Learned and birth revision: fail.
- Donor A/B → host: `address_not_organism_owned` (projection hashes matched; host did not follow donor mapping).
- Wipe/checkpoint/birth-restore on the learned arm: fail, consistent with learned associate not holding.

Later ladder codes were not reached as first-match. This is not a v41 candidate review.
