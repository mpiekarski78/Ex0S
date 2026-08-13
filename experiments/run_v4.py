"""v4: many .md notes in S. Select the matching file; do not dump the folder."""

from __future__ import annotations

import argparse
import json
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
from three_memory.library import DISTRACTOR_MARKERS, all_library_notes, write_notes
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


def _clean(context: str) -> bool:
    return not any(m in context for m in DISTRACTOR_MARKERS)


def make(model, device, facts: Path, *, retrieve_mode: str, policy: str, enabled: bool = True) -> LanguageAgent:
    return LanguageAgent(
        model,
        device,
        store_enabled=enabled,
        retrieve_mode=retrieve_mode,
        retrieve_policy=policy,
        collect_mode="off",
        store=MarkdownStore(facts, enabled=enabled),
    )


def run_v4(ckpt: Path, retrieve_mode: str) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_lm(ckpt, device)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    run_dir = REPO_ROOT / "runs" / f"{stamp}_v4_{retrieve_mode}"
    s_dir = run_dir / "S"
    empty_dir = run_dir / "empty"
    off_dir = run_dir / "off"
    write_notes(s_dir, all_library_notes(include_love=True))
    empty_dir.mkdir(parents=True)
    off_dir.mkdir(parents=True)
    n_files = len(list(s_dir.glob("*.md")))

    empty = make(model, device, empty_dir, retrieve_mode=retrieve_mode, policy="select").probe(PROBE)
    h0 = make(model, device, empty_dir, retrieve_mode=retrieve_mode, policy="select").weight_hash()

    sel = make(model, device, s_dir, retrieve_mode=retrieve_mode, policy="select")
    sel.reset_rho()
    selected = sel.probe(PROBE)

    dump = make(model, device, s_dir, retrieve_mode=retrieve_mode, policy="dump")
    dump.reset_rho()
    dumped = dump.probe(PROBE)

    off = make(model, device, off_dir, retrieve_mode=retrieve_mode, policy="select", enabled=False)
    off_before = off.probe(PROBE)
    off.reset_rho()
    off_after = off.probe(PROBE)

    sel.reset_store()
    after_delete = sel.probe(PROBE)

    weights_ok = sel.weight_hash() == h0 and dump.weight_hash() == h0

    # Reload S from disk with a new agent (ρ empty). Rewrite notes after reset_store wiped them.
    write_notes(s_dir, all_library_notes(include_love=True))
    reloaded = make(model, device, s_dir, retrieve_mode=retrieve_mode, policy="select")
    reloaded.reset_rho()
    love_reloaded = reloaded.probe(PROBE)

    metrics: dict[str, Any] = {
        "checkpoint": str(ckpt),
        "retrieve_mode": retrieve_mode,
        "n_files": n_files,
        "files": sorted(p.name for p in s_dir.glob("*.md")),
        "weight_hash": h0,
        "weights_unchanged_all": weights_ok,
        "empty_prior": _slim(empty),
        "love_S_on_before_rho_reset": _slim(selected),
        "love_S_on_after_rho_reset": _slim(selected),
        "love_S_on_reloaded_from_md": _slim(love_reloaded),
        "love_S_off_before_rho_reset": _slim(off_before),
        "love_S_off_after_rho_reset": _slim(off_after),
        "love_reset_S": _slim(after_delete),
        "dump_all": _slim(dumped),
        "love_has_inspectable_fact": has_love_fact(reloaded),
        "selection_clean": _clean(love_reloaded["context"]),
        "dump_contains_distractors": not _clean(dumped["context"]),
        "selected_prefixes": love_reloaded.get("retrieve", {}).get("prefixes"),
        "dump_n_chosen": dumped.get("retrieve", {}).get("n_chosen"),
        "select_n_rejected": love_reloaded.get("retrieve", {}).get("n_rejected"),
        "NOTE_in_select_context": "NOTE:" in love_reloaded["context"],
        "delta_p_v_select": love_reloaded["p_v"] - empty["p_v"],
        "delta_p_v_dump": dumped["p_v"] - empty["p_v"],
    }
    fake = {
        "weights_unchanged_all": weights_ok,
        "love_S_on_after_rho_reset": {"p_v": love_reloaded["p_v"]},
        "empty_prior": {"p_v": empty["p_v"]},
        "love_S_off_after_rho_reset": {"p_v": off_after["p_v"]},
        "love_S_off_before_rho_reset": {"p_v": off_before["p_v"]},
        "love_reset_S": {"p_v": after_delete["p_v"]},
        "love_has_inspectable_fact": metrics["love_has_inspectable_fact"],
    }
    label, rationale = classify(fake)
    extra = " Select one .md among many; dump-all is a control, not the label."
    if not metrics["selection_clean"]:
        extra += " FAIL selection: distractors leaked into select context."
        if label == "Store-works":
            label, rationale = "Fail", "Select context included clutter; not a use-protocol."
    metrics["classification"] = label
    metrics["rationale"] = rationale + extra
    metrics["run_dir"] = str(run_dir)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(
        f"""# v4 select vs dump ({retrieve_mode})

Classification: **{label}**

{metrics['rationale']}

n_files={n_files}. Select context: `{love_reloaded['context']!r}`
Dump context (truncated): `{dumped['context'][:240]!r}…`

| Check | Result |
|-------|--------|
| Empty prior P(v) | {empty['p_v']:.4f} |
| Select, ρ empty P(v) | {love_reloaded['p_v']:.4f} |
| Dump-all, ρ empty P(v) | {dumped['p_v']:.4f} |
| Select clean (no clutter markers) | {metrics['selection_clean']} |
| Dump contains clutter | {metrics['dump_contains_distractors']} |
| Files rejected by select | {metrics['select_n_rejected']} |
| Dump notes injected | {metrics['dump_n_chosen']} |
| Delete S then probe P(v) | {after_delete['p_v']:.4f} |
| Weights unchanged | {weights_ok} |
""",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="v4: select the right .md among many")
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
        m = run_v4(ckpt, retrieve)
        print(json.dumps({k: m[k] for k in ("classification", "rationale", "run_dir", "retrieve_mode", "selection_clean")}, indent=2))
        print("select P(v)", m["love_S_on_reloaded_from_md"]["p_v"], "ctx", repr(m["love_S_on_reloaded_from_md"]["context"]))
        print("dump P(v)", m["dump_all"]["p_v"], "n_chosen", m["dump_n_chosen"])


if __name__ == "__main__":
    main()
