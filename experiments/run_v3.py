"""v3: S is a folder of .md files. No embeddings. Reload from disk after ρ reset."""

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
from three_memory.bytes_util import LINE_LORD, LINE_LOVE, PROBE
from three_memory.lm_agent import LanguageAgent, probe_js
from three_memory.md_store import MarkdownStore


def _slim(d: dict[str, Any]) -> dict[str, Any]:
    return {
        "p_r": d["p_r"],
        "p_v": d["p_v"],
        "argmax_ch": bytes([d["argmax"]]).decode("latin-1", errors="replace"),
        "context": d["context"],
        "n_store": d["n_store"],
        "rho_l2": d["rho_l2"],
    }


def make(model, device, facts: Path, *, enabled: bool, retrieve_mode: str) -> LanguageAgent:
    store = MarkdownStore(facts, enabled=enabled)
    return LanguageAgent(
        model,
        device,
        store_enabled=enabled,
        retrieve_mode=retrieve_mode,
        store=store,
    )


def run_v3(ckpt: Path, retrieve_mode: str, exposures: int, seed: int) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_lm(ckpt, device)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    run_dir = REPO_ROOT / "runs" / f"{stamp}_v3_{retrieve_mode}"
    run_dir.mkdir(parents=True, exist_ok=True)
    love_dir = run_dir / "facts" / "love"
    lord_dir = run_dir / "facts" / "lord"
    off_dir = run_dir / "facts" / "off"
    reload_dir = run_dir / "facts" / "reload"
    for d in (love_dir, lord_dir, off_dir, reload_dir):
        d.mkdir(parents=True)

    empty_a = make(model, device, run_dir / "facts" / "empty_prior", enabled=True, retrieve_mode=retrieve_mode)
    empty = empty_a.probe(PROBE)

    love = make(model, device, love_dir, enabled=True, retrieve_mode=retrieve_mode)
    h0 = love.weight_hash()
    teach_love = love.experience(LINE_LOVE * exposures)
    files_after_write = love.store.list_files()
    love_before = love.probe(PROBE)
    love.reset_rho()
    love_after_rho = love.probe(PROBE)

    # Cross-instance: new agent, empty ρ, only the .md folder.
    shutil.copytree(love_dir, reload_dir, dirs_exist_ok=True)
    reloaded = make(model, device, reload_dir, enabled=True, retrieve_mode=retrieve_mode)
    reloaded.reset_rho()
    love_reloaded = reloaded.probe(PROBE)
    md_bodies = {
        p.name: (reload_dir / p.name).read_text(encoding="utf-8")
        for p in sorted(reload_dir.glob("*.md"))
    }

    lord = make(model, device, lord_dir, enabled=True, retrieve_mode=retrieve_mode)
    lord.experience(LINE_LORD * exposures)
    lord.reset_rho()
    lord_after = lord.probe(PROBE)

    off = make(model, device, off_dir, enabled=False, retrieve_mode=retrieve_mode)
    off.experience(LINE_LOVE * exposures)
    off_before = off.probe(PROBE)
    off.reset_rho()
    off_after = off.probe(PROBE)
    off_files = list(off_dir.glob("*.md"))

    love.reset_store()
    love_after_delete = love.probe(PROBE)

    weights_ok = love.weight_hash() == h0 and reloaded.weight_hash() == h0 and teach_love["weights_unchanged"]

    metrics: dict[str, Any] = {
        "seed": seed,
        "exposures": exposures,
        "checkpoint": str(ckpt),
        "retrieve_mode": retrieve_mode,
        "store": "markdown",
        "weight_hash": h0,
        "weights_unchanged_all": weights_ok,
        "empty_prior": _slim(empty),
        "love_S_on_before_rho_reset": _slim(love_before),
        "love_S_on_after_rho_reset": _slim(love_after_rho),
        "love_S_on_reloaded_from_md": _slim(love_reloaded),
        "lord_S_on_after_rho_reset": _slim(lord_after),
        "love_S_off_before_rho_reset": _slim(off_before),
        "love_S_off_after_rho_reset": _slim(off_after),
        "love_reset_S": _slim(love_after_delete),
        "love_has_inspectable_fact": has_love_fact(reloaded),
        "files_after_write": files_after_write,
        "files_when_disabled": [p.name for p in off_files],
        "md_bodies": md_bodies,
        "NOTE_in_reload_context": "NOTE:" in love_reloaded["context"],
        "delta_p_v_after_rho_reset": love_after_rho["p_v"] - empty["p_v"],
        "delta_p_v_reloaded": love_reloaded["p_v"] - empty["p_v"],
        "delta_p_v_off_after": off_after["p_v"] - empty["p_v"],
        "js_reload_vs_inprocess": probe_js(love_after_rho, love_reloaded),
        "teach_writes": teach_love["writes"],
    }

    # Classify on the *reloaded* probe: that is the file-persistence test.
    fake = {
        "weights_unchanged_all": weights_ok,
        "love_S_on_after_rho_reset": {"p_v": love_reloaded["p_v"]},
        "empty_prior": {"p_v": empty["p_v"]},
        "love_S_off_after_rho_reset": {"p_v": off_after["p_v"]},
        "love_S_off_before_rho_reset": {"p_v": off_before["p_v"]},
        "love_reset_S": {"p_v": love_after_delete["p_v"]},
        "love_has_inspectable_fact": metrics["love_has_inspectable_fact"],
    }
    label, rationale = classify(fake)
    metrics["classification"] = label
    metrics["rationale"] = rationale + " Store is a folder of .md files; probe after new agent reload."
    metrics["run_dir"] = str(run_dir)

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    bodies_block = "\n\n".join(f"### {name}\n\n```markdown\n{body}```" for name, body in md_bodies.items())
    summary = f"""# v3 markdown store ({retrieve_mode})

Classification: **{label}**

{metrics['rationale']}

Retrieve mode: `{retrieve_mode}`. Files after write: {files_after_write}

Reloaded context: `{love_reloaded['context']!r}`

| Check | Result |
|-------|--------|
| Empty prior P(v) | {empty['p_v']:.4f} |
| After love, before ρ reset P(v) | {love_before['p_v']:.4f} |
| After ρ reset (same process) P(v) | {love_after_rho['p_v']:.4f} |
| **New agent, ρ empty, load .md** P(v) | {love_reloaded['p_v']:.4f} |
| disable-S after ρ reset P(v) | {off_after['p_v']:.4f} |
| Delete .md then probe P(v) | {love_after_delete['p_v']:.4f} |
| NOTE in reloaded context | {metrics['NOTE_in_reload_context']} |
| Files when S disabled | {metrics['files_when_disabled']} |
| Weights unchanged | {weights_ok} |

## Files on disk

{bodies_block}
"""
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="v3: markdown files as S (no RAG)")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--retrieve", choices=("note", "raw"), default="note")
    p.add_argument("--exposures", type=int, default=8)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--both", action="store_true", help="run note+prior.pt and raw+prior_plain.pt")
    args = p.parse_args()

    jobs: list[tuple[Path, str]] = []
    if args.both:
        jobs = [
            (REPO_ROOT / "checkpoints" / "prior.pt", "note"),
            (REPO_ROOT / "checkpoints" / "prior_plain.pt", "raw"),
        ]
    else:
        if args.checkpoint:
            ckpt = Path(args.checkpoint)
        else:
            name = "prior.pt" if args.retrieve == "note" else "prior_plain.pt"
            ckpt = REPO_ROOT / "checkpoints" / name
        jobs = [(ckpt, args.retrieve)]

    for ckpt, retrieve in jobs:
        if not ckpt.is_file():
            raise SystemExit(f"missing {ckpt}")
        m = run_v3(ckpt, retrieve, args.exposures, args.seed)
        print(json.dumps({k: m[k] for k in ("classification", "rationale", "run_dir", "retrieve_mode")}, indent=2))
        print("files", m["files_after_write"])
        print("reloaded P(v)", m["love_S_on_reloaded_from_md"]["p_v"], "ctx", repr(m["love_S_on_reloaded_from_md"]["context"]))
        print("NOTE in ctx", m["NOTE_in_reload_context"])


if __name__ == "__main__":
    main()
