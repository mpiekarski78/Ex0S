# TM.0.29.INDEXING contract

**Lab:** TM.0.29.INDEXING · **Organism:** v36 hippocampal indexing / pattern separation

Fresh `TM029.INDEXING.DEV./TWIN.` worlds. **82 cells.** Standalone runner: do **not** patch TM028 or `gr` module globals.

## Dual path

Write/replacement remains P1 L2 **0.05**. Recall uses stored `key` / `key_rho`. Never key→handle.

## Treatment vs 5-arm ablation

Train once with `act_recall_mode=separated_key`, checkpoint, clone:

| Mode | Role |
|------|------|
| `off` | Necessity of episodic recall |
| `raw_p1` | Verbatim v35 nearest-P1 |
| `early_raw` | Earlier capture without sparse hashing |
| `separated_key_no_familiarity` | Sparse index without 5/8 gate |
| `separated_key` | Treatment |

Each ablation record includes taught-pair **stable** probes **and** novel-cue probes (`path`, `familiar`, retrieved slot). Controls are **observations**, not required failures.

## Novelty (treatment)

Success: episodic path rejected, `path=cortical_fallback`, `familiar=false`. HOLD is not required.

## Perturbation

For separated-key modes: perturb **early live `key_rho`**, re-derive sparse key, retrieve, score. Record retrieved episode identity per trial, not only ACT correctness.

## Decision

Operational first-match on treatment only (`indexing_*_fail`). If treatment passes, interpret controls:

- `indexing_operational_pass__separation_not_causal`
- `sparse_pattern_separation_supported`
- `familiarity_gate_causal`
- `indexing_battery_pass` (operational + sparse vs early_raw + familiarity vs no-familiarity)

Do not name unexpected control success an architectural failure.

## Control

v35/TM028 historical decisions — not rescored. Product **0.0.004**.
