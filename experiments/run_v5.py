"""v5: unread library W. Collect into S (commit) vs peek vs off."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_v1 import classify, has_love_fact
from three_memory.byte_lm import load_lm
from three_memory.bytes_util import PROBE
from three_memory.library import DISTRACTOR_MARKERS, WorldLibrary, all_library_notes, write_notes
from three_memory.lm_agent import LanguageAgent
from three_memory.md_store import MarkdownStore


def _slim(d: dict[str, Any]) -> dict[str, Any]:
    return {
        "p_r": d["p_r"],
        "p_v": d["p_v"],
        "argmax_ch": bytes([d["argmax"]]).decode("latin-1", errors="replace"),
        "context": d["context"],
        "n_store": d["n_store"],
        "rho_l2": d["rho_l2"],
        "retrieve": d.get("retrieve", {}),
    }


def make(
    model,
    device,
    s_dir: Path,
    w_dir: Path | None,
    *,
    retrieve_mode: str,
    collect_mode: str,
    enabled: bool = True,
) -> LanguageAgent:
    world = WorldLibrary(w_dir, enabled=True) if w_dir is not None else None
    return LanguageAgent(
        model,
        device,
        store_enabled=enabled,
        retrieve_mode=retrieve_mode,
        retrieve_policy="select",
        collect_mode=collect_mode,
        store=MarkdownStore(s_dir, enabled=enabled),
        world=world,
    )


def run_v5(ckpt: Path, retrieve_mode: str) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_lm(ckpt, device)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    run_dir = REPO_ROOT / "runs" / f"{stamp}_v5_{retrieve_mode}"
    w_dir = run_dir / "W"
    s_commit = run_dir / "S_commit"
    s_peek = run_dir / "S_peek"
    s_off = run_dir / "S_off"
    s_reload = run_dir / "S_reload"
    empty = run_dir / "empty"
    write_notes(w_dir, all_library_notes(include_love=True))
    for d in (s_commit, s_peek, s_off, s_reload, empty):
        d.mkdir(parents=True)

    prior_a = make(model, device, empty, None, retrieve_mode=retrieve_mode, collect_mode="off")
    h0 = prior_a.weight_hash()
    prior = prior_a.probe(PROBE)

    # Commit: take matching W file into S, then unmount W.
    commit_a = make(model, device, s_commit, w_dir, retrieve_mode=retrieve_mode, collect_mode="commit")
    commit_first = commit_a.probe(PROBE)
    commit_files = commit_a.store.list_files() if hasattr(commit_a.store, "list_files") else []
    commit_a.reset_rho()
    shutil.copytree(s_commit, s_reload, dirs_exist_ok=True)
    committed_only = make(model, device, s_reload, None, retrieve_mode=retrieve_mode, collect_mode="off")
    committed_only.reset_rho()
    commit_unmounted = committed_only.probe(PROBE)
    love_fact = has_love_fact(committed_only)

    # Peek: use W this session, do not write S, then unmount W.
    peek_a = make(model, device, s_peek, w_dir, retrieve_mode=retrieve_mode, collect_mode="peek")
    peek_first = peek_a.probe(PROBE)
    peek_files = peek_a.store.list_files() if hasattr(peek_a.store, "list_files") else []
    peek_a.reset_rho()
    peek_unmounted = make(model, device, s_peek, None, retrieve_mode=retrieve_mode, collect_mode="off")
    peek_unmounted.reset_rho()
    peek_after = peek_unmounted.probe(PROBE)

    # Off: W present, collect disabled.
    off_a = make(model, device, s_off, w_dir, retrieve_mode=retrieve_mode, collect_mode="off")
    off_first = off_a.probe(PROBE)
    off_a.reset_rho()
    off_after = off_a.probe(PROBE)

    committed_only.reset_store()
    after_delete = committed_only.probe(PROBE)

    weights_ok = all(
        ag.weight_hash() == h0
        for ag in (commit_a, peek_a, off_a, committed_only, peek_unmounted)
    )
    selection_clean = not any(m in commit_unmounted["context"] for m in DISTRACTOR_MARKERS)

    metrics: dict[str, Any] = {
        "checkpoint": str(ckpt),
        "retrieve_mode": retrieve_mode,
        "n_world_files": len(list(w_dir.glob("*.md"))),
        "weight_hash": h0,
        "weights_unchanged_all": weights_ok,
        "empty_prior": _slim(prior),
        "commit_first_probe": _slim(commit_first),
        "love_S_on_after_rho_reset": _slim(commit_unmounted),
        "love_S_on_reloaded_from_md": _slim(commit_unmounted),
        "peek_first_probe": _slim(peek_first),
        "peek_after_unmount": _slim(peek_after),
        "love_S_off_before_rho_reset": _slim(off_first),
        "love_S_off_after_rho_reset": _slim(off_after),
        "love_reset_S": _slim(after_delete),
        "love_has_inspectable_fact": love_fact,
        "commit_S_files": commit_files,
        "peek_S_files": peek_files,
        "selection_clean": selection_clean,
        "NOTE_in_commit_context": "NOTE:" in commit_unmounted["context"],
        "w_still_has_love": (w_dir / "my-lo.md").is_file(),
        "delta_p_v_commit_unmounted": commit_unmounted["p_v"] - prior["p_v"],
        "delta_p_v_peek_unmounted": peek_after["p_v"] - prior["p_v"],
        "delta_p_v_off": off_after["p_v"] - prior["p_v"],
    }
    # Classify on commit + unmount W (the long-term collect test).
    fake = {
        "weights_unchanged_all": weights_ok,
        "love_S_on_after_rho_reset": {"p_v": commit_unmounted["p_v"]},
        "empty_prior": {"p_v": prior["p_v"]},
        "love_S_off_after_rho_reset": {"p_v": off_after["p_v"]},
        "love_S_off_before_rho_reset": {"p_v": off_first["p_v"]},
        "love_reset_S": {"p_v": after_delete["p_v"]},
        "love_has_inspectable_fact": metrics["love_has_inspectable_fact"],
    }
    label, rationale = classify(fake)
    metrics["classification"] = label
    metrics["rationale"] = (
        rationale
        + " Collect: commit copies W→S then unmounts W. Peek uses W once and writes nothing."
    )
    metrics["run_dir"] = str(run_dir)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(
        f"""# v5 collect from W ({retrieve_mode})

Classification: **{label}** (on **commit + unmount W**)

{metrics['rationale']}

| Check | Result |
|-------|--------|
| Empty prior P(v) | {prior['p_v']:.4f} |
| Commit, W still mounted P(v) | {commit_first['p_v']:.4f} |
| **Commit, W unmounted, ρ empty** P(v) | {commit_unmounted['p_v']:.4f} |
| Peek, W mounted P(v) | {peek_first['p_v']:.4f} |
| Peek, W unmounted P(v) | {peek_after['p_v']:.4f} |
| Collect off, W mounted P(v) | {off_after['p_v']:.4f} |
| S files after commit | {commit_files} |
| S files after peek | {peek_files} |
| W still has my-lo.md | {metrics['w_still_has_love']} |
| Delete S then probe P(v) | {after_delete['p_v']:.4f} |
| Weights unchanged | {weights_ok} |
""",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="v5: collect from unread library W")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--retrieve", choices=("note", "raw"), default="note")
    p.add_argument("--both", action="store_true")
    args = p.parse_args()
    jobs = (
        [(REPO_ROOT / "checkpoints" / "prior.pt", "note"), (REPO_ROOT / "checkpoints" / "prior_plain.pt", "raw")]
        if args.both
        else [(Path(args.checkpoint) if args.checkpoint else REPO_ROOT / "checkpoints" / ("prior.pt" if args.retrieve == "note" else "prior_plain.pt"), args.retrieve)]
    )
    for ckpt, retrieve in jobs:
        if not ckpt.is_file():
            raise SystemExit(f"missing {ckpt}")
        m = run_v5(ckpt, retrieve)
        print(json.dumps({k: m[k] for k in ("classification", "rationale", "run_dir", "retrieve_mode", "commit_S_files", "peek_S_files")}, indent=2))
        print("commit unmounted P(v)", m["love_S_on_reloaded_from_md"]["p_v"], "ctx", repr(m["love_S_on_reloaded_from_md"]["context"]))
        print("peek unmounted P(v)", m["peek_after_unmount"]["p_v"])
        print("off P(v)", m["love_S_off_after_rho_reset"]["p_v"])


if __name__ == "__main__":
    main()
