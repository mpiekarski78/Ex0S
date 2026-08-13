"""v7: native integer tags, no English prior. v0 world + v5/v6 W/S select/collect."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_v0 import classify, foil_blue_door, probe, teach_red_door
from three_memory.agent import ThreeMemoryAgent
from three_memory.symbols import ACT_USE_KEY, DOOR_RED, RED_FACT_ID
from three_memory.tag_store import TagLibrary, TagStore, all_tag_notes, write_tag_notes


def _run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = REPO_ROOT / "runs" / f"{stamp}_v7"
    d.mkdir(parents=True, exist_ok=True)
    return d


def has_red(agent: ThreeMemoryAgent) -> bool:
    return any(
        r.tags.get("door") == DOOR_RED and r.tags.get("action") == ACT_USE_KEY for r in agent.store.records()
    )


def make(
    s_dir: Path,
    w_dir: Path | None = None,
    *,
    enabled: bool = True,
    collect_mode: str = "off",
    retrieve_policy: str = "select",
) -> ThreeMemoryAgent:
    world = TagLibrary(w_dir) if w_dir is not None else None
    return ThreeMemoryAgent(
        store_enabled=enabled,
        cortex_seed=1337,
        native=True,
        retrieve_policy=retrieve_policy,
        collect_mode=collect_mode,
        store=TagStore(s_dir, enabled=enabled),
        world=world,
    )


def run_v7(seed: int = 12345) -> dict[str, Any]:
    run_dir = _run_dir()
    exp_s = run_dir / "S_exp"
    foil_s = run_dir / "S_foil"
    off_s = run_dir / "S_off"
    w_dir = run_dir / "W"
    s_commit = run_dir / "S_commit"
    s_peek = run_dir / "S_peek"
    s_collect_off = run_dir / "S_collect_off"
    s_reload = run_dir / "S_reload"
    s_dump = run_dir / "S_dump"
    empty = run_dir / "empty"
    write_tag_notes(w_dir, all_tag_notes(include_red=True))
    write_tag_notes(s_dump, all_tag_notes(include_red=True))
    for d in (exp_s, foil_s, off_s, s_commit, s_peek, s_collect_off, s_reload, empty):
        d.mkdir(parents=True, exist_ok=True)

    empty_a = make(empty, None)
    h0 = empty_a.weight_hash()
    prior = probe(empty_a, "probe_red_with_key", seed=seed + 10)

    # Experience → tag files on disk (no English what).
    A = make(exp_s, None, collect_mode="off")
    teach_A = teach_red_door(A, seed=seed)
    probe_A_before = probe(A, "probe_red_with_key", seed=seed + 10)
    rho_A = A.rho.snapshot()
    A.reset_rho()
    probe_A_after = probe(A, "probe_red_with_key", seed=seed + 10)
    A.rho.load(rho_A)
    probe_A_restored = probe(A, "probe_red_with_key", seed=seed + 10)
    A.reset_rho()
    exp_files = A.store.list_files() if hasattr(A.store, "list_files") else []

    B = make(foil_s, None)
    teach_B = foil_blue_door(B, seed=seed + 1)
    B.reset_rho()
    probe_B_after = probe(B, "probe_red_with_key", seed=seed + 10)

    C = make(off_s, None, enabled=False)
    teach_C = teach_red_door(C, seed=seed)
    probe_C_before = probe(C, "probe_red_with_key", seed=seed + 10)
    C.reset_rho()
    probe_C_after = probe(C, "probe_red_with_key", seed=seed + 10)

    A2 = make(run_dir / "S_reset", None)
    teach_red_door(A2, seed=seed)
    A2.reset_rho()
    A2.reset_store()
    probe_reset_S = probe(A2, "probe_red_with_key", seed=seed + 10)

    # New agent, empty ρ, only the experience .tag folder.
    reloaded = make(exp_s, None)
    reloaded.reset_rho()
    probe_reload = probe(reloaded, "probe_red_with_key", seed=seed + 10)

    # Collect from unread W.
    commit_a = make(s_commit, w_dir, collect_mode="commit")
    probe_commit_first = probe(commit_a, "probe_red_with_key", seed=seed + 10)
    commit_files = commit_a.store.list_files() if hasattr(commit_a.store, "list_files") else []
    shutil.copytree(s_commit, s_reload, dirs_exist_ok=True)
    committed_only = make(s_reload, None, collect_mode="off")
    committed_only.reset_rho()
    probe_commit_unmount = probe(committed_only, "probe_red_with_key", seed=seed + 10)

    peek_a = make(s_peek, w_dir, collect_mode="peek")
    probe_peek_first = probe(peek_a, "probe_red_with_key", seed=seed + 10)
    peek_files = peek_a.store.list_files() if hasattr(peek_a.store, "list_files") else []
    peek_unmounted = make(s_peek, None, collect_mode="off")
    probe_peek_unmount = probe(peek_unmounted, "probe_red_with_key", seed=seed + 10)

    off_w = make(s_collect_off, w_dir, collect_mode="off")
    probe_w_off = probe(off_w, "probe_red_with_key", seed=seed + 10)

    dump_a = make(s_dump, None, retrieve_policy="dump")
    dump_a.reset_rho()
    probe_dump = probe(dump_a, "probe_red_with_key", seed=seed + 10)

    weights_ok = all(
        ag.weight_hash() == h0
        for ag in (A, B, C, reloaded, commit_a, peek_a, dump_a, committed_only)
    ) and teach_A["weights_unchanged"] and teach_B["weights_unchanged"] and teach_C["weights_unchanged"]

    tag_bodies = {p.name: p.read_text(encoding="utf-8") for p in sorted(exp_s.glob("*.tag"))}
    commit_whats = [r.what for r in commit_a.store.records()]
    english_in_tags = any(
        w in (body + " " + " ".join(commit_whats)).lower()
        for body in tag_bodies.values()
        for w in ("door opens", "only with", "love", "lord", "note:")
    )

    metrics: dict[str, Any] = {
        "seed": seed,
        "native": True,
        "alphabet": "integer tags + bit observations",
        "genome": "frozen cortex seed 1337 (not ACGT)",
        "weight_hash": h0,
        "weights_unchanged_all": weights_ok,
        "empty_prior_action": prior["action_name"],
        "empty_prior_correct": prior["correct"],
        "A_has_inspectable_fact": has_red(A),
        "A_rho_l2_after_exp": teach_A["rho_l2"],
        "A_probe_before_rho_reset": probe_A_before,
        "A_probe_after_rho_reset": probe_A_after,
        "A_probe_after_rho_restore": probe_A_restored,
        "B_probe_after_rho_reset": probe_B_after,
        "disable_S_probe_before_rho_reset": probe_C_before,
        "disable_S_probe_after_rho_reset": probe_C_after,
        "reset_S_probe_after_rho_reset": probe_reset_S,
        "reload_from_tags": probe_reload,
        "commit_first": probe_commit_first,
        "commit_unmount": probe_commit_unmount,
        "peek_first": probe_peek_first,
        "peek_unmount": probe_peek_unmount,
        "collect_off": probe_w_off,
        "dump_all": probe_dump,
        "exp_S_files": exp_files,
        "commit_S_files": commit_files,
        "peek_S_files": peek_files,
        "tag_bodies": tag_bodies,
        "english_prose_in_tags": english_in_tags,
        "n_world_files": len(list(w_dir.glob("*.tag"))),
        "twin_rho_distance": {"l2": 0.0},
        "rho_restore_action_match": probe_A_before["action"] == probe_A_restored["action"],
    }
    label, rationale = classify(metrics)
    # Classify on experience+reload as the native-S test; require collect-unmount too for Store-works.
    if label == "Store-works" and not probe_reload["correct"]:
        label, rationale = "Fail", "Tag files did not reload into a correct probe."
    if label == "Store-works" and not probe_commit_unmount["correct"]:
        label, rationale = "Fail", "Collect commit did not survive unmounting W."
    if label == "Store-works" and probe_peek_unmount["correct"]:
        label, rationale = "Fail", "Peek after unmount still correct; W was treated as memory."
    if english_in_tags:
        label, rationale = "Fail", "Tag files contained English prose (illegal in v7)."
    metrics["classification"] = label
    metrics["rationale"] = rationale + " v7 native integer tags; no English prior; genome=cortex seed."
    metrics["run_dir"] = str(run_dir)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(
        f"""# v7 native tags (no English)

Classification: **{label}**

{metrics['rationale']}

Empty prior action: `{prior['action_name']}` (correct={prior['correct']}).
Experience files: {exp_files}. Collect S: {commit_files}.

| Check | Result |
|-------|--------|
| A after ρ reset | {probe_A_after['correct']} ({probe_A_after['action_name']}) |
| Reload .tag files, ρ empty | {probe_reload['correct']} ({probe_reload['action_name']}) |
| B foil after ρ reset | {probe_B_after['correct']} ({probe_B_after['action_name']}) |
| disable-S after ρ reset | {probe_C_after['correct']} ({probe_C_after['action_name']}) |
| Collect commit, unmount W | {probe_commit_unmount['correct']} ({probe_commit_unmount['action_name']}) |
| Peek then unmount W | {probe_peek_unmount['correct']} ({probe_peek_unmount['action_name']}) |
| Collect off, W mounted | {probe_w_off['correct']} ({probe_w_off['action_name']}) |
| Dump-all tags | {probe_dump['correct']} ({probe_dump['action_name']}) |
| Delete S | {probe_reset_S['correct']} |
| English prose in tags | {english_in_tags} |
| Weights unchanged | {weights_ok} |

## Experience S

```text
{json.dumps(tag_bodies, indent=2)}
```
""",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="v7 native integer tags, no English prior")
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()
    m = run_v7(seed=args.seed)
    print(json.dumps({k: m[k] for k in ("classification", "rationale", "run_dir")}, indent=2))
    print("prior", m["empty_prior_action"], "reload", m["reload_from_tags"]["correct"], m["reload_from_tags"]["action_name"])
    print("commit unmount", m["commit_unmount"]["correct"], "peek unmount", m["peek_unmount"]["correct"])
    print("dump", m["dump_all"]["correct"], m["dump_all"]["action_name"])
    print("files", m["exp_S_files"], m["commit_S_files"])


if __name__ == "__main__":
    main()
