# TM.0.37.JOINTPROJ contract

**Lab:** TM.0.37.JOINTPROJ · **Organism:** v39 (neural after this freeze)

Fresh `TM037.JOINTPROJ.DEV./TWIN.` worlds. Do **not** edit TM031–TM036 locks. Do **not** change rehearsal cap, write L2, `R`, or `ACT_RECALL_MODES`. Do **not** install oracle \(W^*\) as organism law.

## Why this wall

TM036: native sequential updates and Jacobi fail; protect-every-row (GEM-like) retains established rows but does not acquire; a nearby feasible \(W^*\) restores store 0 and live 8/8. The organism does not need more capacity or replay. It needs a **joint minimal-change consolidator**.

## Convexity (prerequisite, not a slogan)

Query scores: \(q=W p_1\), cosine against unit motor vectors.

- **Ranking** unique-winner \(\Leftrightarrow (v_h-v_r)\cdot(W p_1)>0\). Common \(|q|\) cancels. This **is** a Euclidean halfspace in \(W\): \(\langle W,\mathrm{outer}(d,p_1)\rangle_F>0\).
- **Geometric \(\gamma\)** is \(\gamma=(W^\top d)\cdot p_1/\|W^\top d\|\), scale-invariant in \(W\). The floor \(\gamma\ge\tau\) with \(\tau=\) `ACT_MARGIN_FLOOR` \(=0.01\) is a **second-order cone**, not a halfspace.

Ordinary Passive-Aggressive / Dykstra therefore **cannot** be claimed on geometric \(\gamma\) itself. Each cycle freezes \(b=\tau\|W_{\mathrm{ref}}^\top d\|\) and projects onto the supporting halfspace \(\langle W,\mathrm{outer}(d,x)\rangle_F\ge b\). After the cycle, recompute \(b\). **Stop on the existing nonlinear** `_episode_rehearsal_violation`. That is linearized supporting-halfspace PA, not “\(\gamma\) is a halfspace.”

## Parent

Teach credits **0–6** with default v37. Apply TM035 `write_only` for credit 7. Checkpoint. Every arm clones from that identical post-write snapshot (same parent as TM036).

## Arms (matched)

- `v37` — ranking-error one-shot, then unchanged v37 sequential burst.
- `pa_cyclic` — 16 cycles through **all** valid \((\text{episode},\text{rival})\) supporting halfspaces; analytic PA-I; no Dykstra corrections.
- `dykstra` — same cycle, retaining per-constraint correction terms \(I_i\).
- `oracle` — TM036 closest feasible \(W^*\) (Frobenius). **Diagnostic ceiling only.** Runner-side. Never installed in `neural_cortex`.

No fitted learning rate. Clip `mix_slow=False` at **end of pass**, not after every constraint. Episode handles participate only in consolidation constraints, never ACT routing. REST, half-spacing, recall, and write lifecycle stay frozen.

Fix gate: store violations 0, live ranking 8/8. Margin is recorded, not a live gate.

## Routes (first-match on diagnostic worlds)

Diagnostic = `v37` fails the fix gate.

1. `jointproj_no_v37_fail` — every world already fixes under v37
2. `jointproj_pa_sufficient` — PA cyclic fixes every diagnostic world (Dykstra may also)
3. `jointproj_dykstra_required` — only Dykstra-corrected projection fixes every diagnostic world
4. `jointproj_oracle_only` — only \(W^*\) fixes
5. `jointproj_capacity_wall` — oracle itself fails stored-row margins
6. `jointproj_mixed_routes` — diagnostic worlds disagree

**No `cortex.candidate.v39.lock` from this lab.** A later candidate decision is separate. Product **0.0.004**.
