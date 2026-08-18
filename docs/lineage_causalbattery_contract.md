# TM.0.40.CAUSALBATTERY contract

**Lab:** TM.0.40.CAUSALBATTERY · **Not a v40 candidate from this freeze.**

Fresh `TM040.CAUSAL.DEV./TWIN.` worlds. Frozen organism (v37 credit + optional joint SOCP). Do **not** edit `neural_cortex.py` or `joint_socp.py`. Do **not** rewrite TM039 locks. Do **not** write `cortex.candidate.v40.lock`.

## Narrow claim already earned (TM039)

When frozen v37 leaves jointly feasible stored constraints unresolved, an atomic minimum-change SOCP can restore them. That is architecture-level rescue on the diagnosed snapshot, not general lifelong learning.

## This battery

`fallback_joint` is the scored arm. `always_joint` is observational. `v37` is the matched local-plasticity control.

Required `fallback_joint` kinds: acquisition, stability (includes perturbation), history, novelty, reversal, specificity, 8×4, later-learning.

- **Causal rescue:** a cell stem where `v37` fails and `fallback_joint` succeeds. Count separately.
- **Compatibility:** untouched cells where fallback never invokes the solver. Not causal efficacy.
- **If no untouched cell activates the solver:** first-match `jointsocp_generalization_not_exercised`. Not a candidate pass.
- **Later-learning** probes the same organism after an installed SOCP, never a reset checkpoint. If SOCP never installed before those probes, later-learning was not exercised.
- **Reversal** is ecological (TM027). A separate `contradict` cell injects coexisting opposite ranking constraints; if the SOCP is infeasible/rejected, \(W\) must be **byte-identical**.
- **Hashes are diagnostic.** Semantic gates are store violations, live ranking, and \(\gamma\ge\tau\). Numerical solver output may vary slightly under pinned versions.

## Observational always_joint

- both pass, fallback moves less / runs less → scheduling efficiency
- fallback passes and always-joint later fails → scheduling is behaviorally causal
- both fail differently → optimizer does not generalize

## Explicit reject laws (tests, not DEV fitted)

Infeasible, non-optimal status, excessive residual, NaN/Inf, organism predicate failure, unavailable solver: reject the entire candidate \(W\); no partial mutation.

Product **0.0.004**.
