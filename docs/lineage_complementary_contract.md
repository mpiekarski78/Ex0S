# TM.0.28.COMPLEMENTARY contract

**Lab:** TM.0.28.COMPLEMENTARY · **Organism:** v35 episodic ACT recall

Fresh `TM028.COMPLEMENTARY.DEV./TWIN.` worlds. **66 cells.**

## Treatment vs ablation

- **Treatment:** `episodic_act_recall=True` on all non-ablation cells.
- **Ablation:** train once → checkpoint → matched clone with recall **OFF**; must fail stable gate.
- Separate flags: `behavioral_pass` (treatment) vs `expected_ablation_failure`.

## Motor path

All probes use `actuator_decision_scores(live_p1)`. Perturbation: perturb live P1 → retrieve → complete → score.

## Novel cue

Never-taught symbol after full training. Confident wrong ACT → `episodic_overgeneralization`.

## Control

v34/R2 historical decisions — not rescored on TM028.
