# TM.0.31.HALFSPACE contract

**Lab:** TM.0.31.HALFSPACE · **Organism:** v37 early-raw + half-minimum-spacing familiarity

Fresh `TM031.HALFSPACE.DEV./TWIN.` worlds. **148 cells.** Four `seed_registry` replicates (seed-only variants). Standalone runner: do **not** patch TM028/TM029/TM030.

## Freeze order

Runner SHA pinned and pushed **before** the neural gate exists. The frozen runner may fail until `early_raw_half_spacing` is implemented. Do not edit the runner after freeze push.

## Treatment

`act_recall_mode=early_raw_half_spacing`. Write stays P1 L2 **0.05**. No Hadamard on the treatment path.

## Novelty success

`familiar=false` and `path=cortical_fallback`. HOLD is not required.

## Causal novel protocol (per skip)

trained checkpoint → clone → allocate pinned dummy `_vocab_vec`s → clone again → gate ON and gate OFF on that second clone. ON/OFF `key_rho` hashes must match before retrieval.

## Controls (observations)

Matched retrieval-path clones on treatment-trained state. `separated_key` is **not** a v36 train reproduction; historical TM029 is the failed-v36 control.

## Decision

Operational first-match on treatment only. Gate is causal only if matched ON/OFF novel queries change retrieval path and treatment novelty succeeds.

Product **0.0.004**.
