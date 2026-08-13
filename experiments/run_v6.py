"""v6: use-skill on the plain prior. No NOTE-copy in weights. Fact stays in the file."""

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
from three_memory.bytes_util import BANNED, PROBE
from three_memory.library import FEWSHOT_DEMOS, TOOL_BYTE_BIAS, WorldLibrary, all_library_notes, write_notes
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


def run_v6(ckpt: Path, retrieve_mode: str) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_lm(ckpt, device)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    run_dir = REPO_ROOT / "runs" / f"{stamp}_v6_{retrieve_mode}"
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

    commit_a = make(model, device, s_commit, w_dir, retrieve_mode=retrieve_mode, collect_mode="commit")
    commit_first = commit_a.probe(PROBE)
    commit_files = commit_a.store.list_files() if hasattr(commit_a.store, "list_files") else []
    commit_a.reset_rho()
    shutil.copytree(s_commit, s_reload, dirs_exist_ok=True)
    committed_only = make(model, device, s_reload, None, retrieve_mode=retrieve_mode, collect_mode="off")
    committed_only.reset_rho()
    commit_unmounted = committed_only.probe(PROBE)
    love_fact = has_love_fact(committed_only)

    peek_a = make(model, device, s_peek, w_dir, retrieve_mode=retrieve_mode, collect_mode="peek")
    peek_first = peek_a.probe(PROBE)
    peek_files = peek_a.store.list_files() if hasattr(peek_a.store, "list_files") else []
    peek_unmounted = make(model, device, s_peek, None, retrieve_mode=retrieve_mode, collect_mode="off")
    peek_after = peek_unmounted.probe(PROBE)

    off_a = make(model, device, s_off, w_dir, retrieve_mode=retrieve_mode, collect_mode="off")
    off_first = off_a.probe(PROBE)
    off_a.reset_rho()
    off_after = off_a.probe(PROBE)

    committed_only.reset_store()
    after_delete = committed_only.probe(PROBE)

    weights_ok = all(ag.weight_hash() == h0 for ag in (commit_a, peek_a, off_a, committed_only, peek_unmounted))
    demo_clean = not any(b in FEWSHOT_DEMOS for b in BANNED)

    metrics: dict[str, Any] = {
        "checkpoint": str(ckpt),
        "retrieve_mode": retrieve_mode,
        "tool_byte_bias": TOOL_BYTE_BIAS,
        "fewshot_demos": FEWSHOT_DEMOS,
        "fewshot_demos_clean": demo_clean,
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
        "NOTE_in_commit_context": "NOTE:" in commit_unmounted["context"],
        "love_in_commit_context": "love" in commit_unmounted["context"],
        "delta_p_v_commit_unmounted": commit_unmounted["p_v"] - prior["p_v"],
        "delta_p_v_peek_unmounted": peek_after["p_v"] - prior["p_v"],
        "delta_p_v_off": off_after["p_v"] - prior["p_v"],
    }
    fake = {
        "weights_unchanged_all": weights_ok,
        "love_S_on_after_rho_reset": {"p_v": commit_unmounted["p_v"]},
        "empty_prior": {"p_v": prior["p_v"]},
        "love_S_off_after_rho_reset": {"p_v": off_after["p_v"]},
        "love_S_off_before_rho_reset": {"p_v": off_first["p_v"]},
        "love_reset_S": {"p_v": after_delete["p_v"]},
        "love_has_inspectable_fact": love_fact,
    }
    label, rationale = classify(fake)
    metrics["classification"] = label
    metrics["rationale"] = (
        rationale
        + f" v6 retrieve={retrieve_mode} on plain prior (no NOTE-copy in weights)."
        + " Classify on commit + unmount W."
    )
    metrics["run_dir"] = str(run_dir)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(
        f"""# v6 use-skill ({retrieve_mode})

Classification: **{label}**

{metrics['rationale']}

Checkpoint: `{ckpt.name}`. Tool bias={TOOL_BYTE_BIAS}. Context: `{commit_unmounted['context']!r}`

| Check | Result |
|-------|--------|
| Empty prior P(v) | {prior['p_v']:.4f} |
| Commit + unmount W, ρ empty P(v) | {commit_unmounted['p_v']:.4f} |
| Peek then unmount P(v) | {peek_after['p_v']:.4f} |
| Collect off P(v) | {off_after['p_v']:.4f} |
| Delete S P(v) | {after_delete['p_v']:.4f} |
| NOTE in context | {metrics['NOTE_in_commit_context']} |
| `love` in context | {metrics['love_in_commit_context']} |
| S files after commit | {commit_files} |
| Weights unchanged | {weights_ok} |
""",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="v6: use-skill on plain prior, no NOTE in weights")
    p.add_argument("--checkpoint", type=str, default=str(REPO_ROOT / "checkpoints" / "prior_plain.pt"))
    p.add_argument("--retrieve", choices=("tool", "fewshot", "note", "raw"), default="tool")
    p.add_argument("--all-modes", action="store_true")
    args = p.parse_args()
    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        raise SystemExit(f"missing {ckpt}; train with: python -m experiments.train_prior --plain")
    modes = ("tool", "fewshot", "note") if args.all_modes else (args.retrieve,)
    for retrieve in modes:
        m = run_v6(ckpt, retrieve)
        print(
            json.dumps(
                {
                    k: m[k]
                    for k in (
                        "classification",
                        "rationale",
                        "run_dir",
                        "retrieve_mode",
                        "delta_p_v_commit_unmounted",
                    )
                },
                indent=2,
            )
        )
        print(
            "commit unmounted P(v)",
            m["love_S_on_reloaded_from_md"]["p_v"],
            "ctx",
            repr(m["love_S_on_reloaded_from_md"]["context"]),
        )


if __name__ == "__main__":
    main()
