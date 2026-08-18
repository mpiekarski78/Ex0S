# TM.0.35.CREDITSPLIT results

Credit-7 write / one-shot / burst / complete split after TM034. Product **0.0.004**.

## Decision: `creditsplit_mixed_oneshot_and_burst`

Both diagnostic worlds (`reg1`, both orders) rebreak previously correct rows under `complete`, and **both the one-shot and the v37 burst independently cause that rebreak**. Write does not. v39 one-shot-only safe scaling is **not** opened.

No neural edit. v38 unchanged. 16-pass cap unchanged. 44 not installed.

- worlds: 8 diagnostic 2 no-rebreak 6
- diagnostic routes: both `mixed_oneshot_and_burst`
- v39 gate open: **false**

git_head `7e2b509d…` (freeze). Frozen runner SHA `52a5f32c…`. Clean tree at DEV start.

## Attribution (new slot excluded)

Previously correct = non-violating stored rows on the pre-credit-7 checkpoint. The newly written slot is excluded. Parent store is at **zero** on all eight worlds.

Write-only leaves protected rows intact on every world (`viol=1` is the new row, not a protected rebreak).

| world | write | oneshot | burst | complete |
|-------|-------|---------|-------|----------|
| reg0 both orders | protected intact | breaks 0/2/4/6 | intact (live 8/8) | intact (burst repairs; 13 updates) |
| **reg1 both orders** | protected intact | **breaks 0/2/4/6** | **breaks 0/4/6** (117 updates, live 6/8) | **breaks 0/4/6** (122 updates, live 6/8) |
| reg2 both orders | protected intact | breaks 0/2/4/6 | intact (live 8/8) | intact (burst repairs; 7 updates) |
| reg3 both orders | intact | intact | intact | intact |

Diagnostic leftover after complete matches TM034: slots **0, 4, 6**. Burst-only (no one-shot) already produces that leftover. One-shot-only produces a larger even-slot collapse that the burst sometimes repairs (reg0/reg2) and sometimes does not (reg1).

## Mechanism routing

Write is not the lever. One-shot-only v39 cannot cover the diagnostic: suppressing or scaling the one-shot still leaves a burst that independently rebreaks protected rows. Mixed one-shot **and** burst causality. Product **0.0.004**.
