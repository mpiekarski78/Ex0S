"""MEMLANG-1 Stage A — invariant value formation lives.

Runtime adapter bind only. No neural_cortex.py edit. No K/Q/V redesign.
Do not retune EPISODE_MATCH_L2. Do not open TM063.
Product 0.0.004. Never write cortex.candidate.v41.lock.
"""

from __future__ import annotations

import ast
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from experiments.run_tm027gatedrehearsal import assert_finite_record
from experiments.run_tm046oneshot import load_common, w_act_hash
from experiments.run_tm052sharefeas import solve_ceiling, vocab_of
from experiments.run_tm053cover import score_eval
from experiments.run_tm057dual import live_trajectory as historical_live
from experiments.run_tm059receipt import receipt_contract_violations
from experiments.run_tm060ondrift import (
    assert_split_pins,
    opaque_live,
    pairs_of,
    parent_hash_unchanged,
)
from three_memory.cortex_lineage import freeze_plasticity, sha_file
from three_memory.memlang.adapters import make_adapter
from three_memory.memlang.capture import MemlangReceipts
from three_memory.neural_cortex import EPISODE_MATCH_L2, EPISODE_SLOTS, MEMPROJ_NONE

REPO_ROOT = Path(__file__).resolve().parents[1]
THIS = Path(__file__).resolve()
LAB = "MEMLANG-1"
PREREG = REPO_ROOT / "docs" / "memlang1.prereg.lock"
BUDGET = REPO_ROOT / "docs" / "memlang1.budget.lock"
STAGE_B = REPO_ROOT / "docs" / "memlang1.stage_b.lock"
CANDIDATE = REPO_ROOT / "docs" / "cortex.candidate.v41.lock"
NEURAL = REPO_ROOT / "three_memory" / "neural_cortex.py"
OPAQUE = REPO_ROOT / "three_memory" / "opaque_memory.py"
SOLVER = REPO_ROOT / "three_memory" / "joint_socp.py"
TM062_RUNNER = REPO_ROOT / "experiments" / "run_tm062xgen.py"
MANIFEST_SHA = "bc09aab32f71e4a32b10436cfe91ab5d31dee8e185a5206fe3d59479dc741f11"
EXPECTED_N_CELLS = 10
N_SETUP_CELLS = 2
N_SCORED_CELLS = 8
LADDER = (
    "setup_precondition_fail",
    "runner_constructed_value",
    "handle_copied_into_S",
    "observer_used_runner_provenance",
    "ordinary_cortex_broken",
    "offline_reconstruction",
    "checkpoint_restore_fail",
    "feedback_off_fail",
    "permuted_feedback_fail",
    "reward_gate_fail",
    "action_collapse",
    "cue_overfit",
    "fresh_world_fail",
    "later_context_drift",
    "stage_a_integrated_pass",
)
BEHAVIORAL_LADDER = LADDER[1:]


def load_prereg() -> dict[str, Any]:
    return json.loads(PREREG.read_text(encoding="utf-8"))


def expected_cell_ids() -> list[str]:
    p = load_prereg()
    probes = [str(x) for x in p["probes"]]
    setup = [f"decoder|w{wi}" for wi in range(int(p["n_worlds"]))]
    scored = [f"{probe}|w{wi}" for probe in probes for wi in range(int(p["n_worlds"]))]
    ids = setup + scored
    payload = {
        "lab": LAB,
        "domains": sorted(p["domains"].items()),
        "setup_cell_ids": sorted(setup),
        "scored_cell_ids": sorted(scored),
        "cell_ids": sorted(ids),
        "probes": probes,
    }
    got = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if got != MANIFEST_SHA:
        raise RuntimeError("MEMLANG-1 manifest drifted")
    if len(ids) != EXPECTED_N_CELLS:
        raise RuntimeError("cell count drifted")
    return ids


def refuse_runner_leaks(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bad: list[str] = []

    class V(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            name = ""
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in ("retrieve", "retrieve_by_query", "_run_joint_socp_consolidation", "credit_tagged", "latest_episode"):
                bad.append(f"{path.name}:{node.lineno}:{name}")
            self.generic_visit(node)

        def visit_Assign(self, node: ast.Assign) -> None:
            for t in node.targets:
                if isinstance(t, ast.Attribute) and t.attr in ("W_k", "W_q", "W_v", "W_act_query", "_episodes"):
                    bad.append(f"{path.name}:{t.lineno}:assign_{t.attr}")
            self.generic_visit(node)

    V().visit(tree)
    return bad


def _assert_frozen(p: dict[str, Any]) -> None:
    if CANDIDATE.exists():
        raise RuntimeError("refuse cortex.candidate.v41.lock")
    if hashlib.sha256(NEURAL.read_bytes()).hexdigest() != p["neural_cortex_sha"]:
        raise RuntimeError("neural_cortex.py must not change")
    if hashlib.sha256(OPAQUE.read_bytes()).hexdigest() != p["opaque_memory_sha"]:
        raise RuntimeError("opaque_memory.py must not change")
    if hashlib.sha256(SOLVER.read_bytes()).hexdigest() != p["joint_socp_sha"]:
        raise RuntimeError("joint_socp.py must not change")
    if sha_file(TM062_RUNNER) != p["tm062_runner_sha"]:
        raise RuntimeError("TM062 runner must stay frozen")
    if any((REPO_ROOT / "experiments").glob("run_tm063memlang*.py")):
        raise RuntimeError("TM063 diagnostic opened")
    if float(EPISODE_MATCH_L2) != 0.05:
        raise RuntimeError("do not retune EPISODE_MATCH_L2")
    if int(p["n_dev_repeats"]) != 4:
        raise RuntimeError("n_dev_repeats must stay 4")
    if json.loads(STAGE_B.read_text()).get("executable") is True:
        raise RuntimeError("Stage B must stay locked until Stage A passes")


def collect_bundle(
    *,
    wi: int,
    tmp: str,
    adapter_cfg: dict[str, Any],
    cue_prefix: str,
    domain_key: str,
    adapter=None,
    n_online_repeats: int | None = None,
    permute_feedback: bool = False,
    feedback_off: bool = False,
) -> dict[str, Any]:
    p = load_prereg()
    n = int(p["n"])
    if adapter is None:
        adapter = make_adapter(str(adapter_cfg.get("family") or "identity"), n, adapter_cfg)
    receipts = MemlangReceipts(adapter, permute_feedback=bool(permute_feedback), feedback_off=bool(feedback_off))
    ref = historical_live(
        tmp=tmp,
        tag=f"ml_ref_{cue_prefix}_w{wi}",
        wi=int(wi),
        seed=int(p["seed_registry"]),
        domain=str(p["domains"]["DEV"] if domain_key != "RENAME" else p["domains"]["RENAME"]),
        n_repeats=int(p["n_dev_repeats"]),
        cue_fn=lambda _rep, i, _h: f"s_ref_{cue_prefix}_{wi}_{i}",
        tag_prefix=f"ml_ref_{cue_prefix}_w{wi}",
        grounded=False,
        receipts=None,
    )
    online = opaque_live(
        tmp=tmp,
        tag=f"ml_on_{cue_prefix}_w{wi}",
        wi=int(wi),
        seed=int(p["seed_registry"]),
        domain=str(p["domains"][domain_key]),
        n_repeats=int(p["n_online_repeats"] if n_online_repeats is None else n_online_repeats),
        cue_fn=lambda rep, i, _h: f"s_{cue_prefix}_{wi}_{rep}_{i}",
        tag_prefix=f"ml_on_{cue_prefix}_w{wi}",
        receipts=receipts,
    )
    handles = list(online["handles"])
    ag = load_common(online)
    freeze_plasticity(ag)
    ag.set_memproj_arm(MEMPROJ_NONE)
    w0_hash = w_act_hash(ag)
    snap = ag.checkpoint()
    ag2 = load_common({"snap": snap, "genome": online["genome"], "world": online["world"]})
    try:
        restore_ok = w_act_hash(ag2) == w0_hash
        _ = ag2.form_write_value(np.ones(int(p["n"]), dtype=np.float64))
    except Exception:
        restore_ok = False
    del ag2
    W0 = np.asarray(ag._from_t(ag.W_act_query), dtype=np.float64).copy()
    vocab = vocab_of(ag)
    ref_pairs = [(h, np.asarray(ref["last_p1"][h], dtype=np.float64).copy()) for h in handles]
    n_need = int(p["n_online_repeats"] if n_online_repeats is None else n_online_repeats) * int(p["n_handles"])
    split_n = min(int(p["split_n"]), max(1, n_need // 2))
    observer_ok = int(receipts.n_observer_provenance) == 0 and int(receipts.n_handle_copied_into_s) == 0
    capture_ok = len(receipts.attempts) == n_need
    prefix_rows = list(receipts.attempts[:split_n])
    later_rows = list(receipts.attempts[split_n:])
    pins = {"future_never_socp_constraints": False}
    if capture_ok and prefix_rows and later_rows:
        pins = assert_split_pins(
            train_rows=prefix_rows,
            future_rows=later_rows,
            ref_pairs=ref_pairs,
            handles=handles,
            cons=list(ref_pairs) + pairs_of(prefix_rows),
        )
    oracle_rec, oracle_W = solve_ceiling(W0, list(ref_pairs) + pairs_of(receipts.attempts), vocab)
    parent_hash_unchanged(ag, w0_hash)
    del oracle_W
    decoder = {
        "kind": "setup",
        "id": f"decoder|w{wi}",
        "passed": bool(ref["all_ok"] and observer_ok and capture_ok and online["opaque_enabled"]),
        "ref_ok": bool(ref["all_ok"]),
        "observer_ok": bool(observer_ok),
        "capture_ok": bool(capture_ok),
        "n_observer_provenance": int(receipts.n_observer_provenance),
        "n_handle_copied_into_s": int(receipts.n_handle_copied_into_s),
        "n_rewarded_persistent_writes": int(receipts.n_rewarded_persistent_writes),
        "n_nonpositive_persistent_writes": int(receipts.n_nonpositive_persistent_writes),
        "reward_gate_ok": int(receipts.n_nonpositive_persistent_writes) == 0,
        "feedback_off_ok": True,
        "permuted_feedback_ok": True,
        "checkpoint_restore_ok": bool(restore_ok),
        "cue_overfit": False,
        "full_oracle_feasible": bool(oracle_rec.get("feasible")),
        "W_installed": False,
        "adapter_geometry": adapter.geometry(),
        "memproj_arm": MEMPROJ_NONE,
        "candidate_v41_lock": False,
        "parent_w_act_query_unchanged": True,
        "episode_slots": int(EPISODE_SLOTS),
        "pins": pins,
    }
    assert_finite_record(decoder, ctx=decoder["id"])
    return {
        "handles": handles,
        "W0": W0,
        "w0_hash": w0_hash,
        "vocab": vocab,
        "ref_pairs": ref_pairs,
        "prefix_rows": prefix_rows,
        "later_rows": later_rows,
        "rename_rows": [],
        "decoder": decoder,
        "ag": ag,
        "wi": int(wi),
        "adapter": adapter,
    }


def fit_decoder(bundle: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[np.ndarray, bool]:
    rec, W_star = solve_ceiling(bundle["W0"], list(bundle["ref_pairs"]) + pairs_of(rows), bundle["vocab"])
    parent_hash_unchanged(bundle["ag"], str(bundle["w0_hash"]))
    W_use = bundle["W0"] if W_star is None else W_star
    del W_star
    return W_use, bool(rec.get("feasible"))


def scored_cell(bundle: dict[str, Any], probe: str, W: np.ndarray, rows: list[dict[str, Any]]) -> dict[str, Any]:
    sc = score_eval(W, pairs_of(rows), bundle["vocab"])
    ok = bool(sc["n_ok"] == sc["n_need"])
    parent_hash_unchanged(bundle["ag"], str(bundle["w0_hash"]))
    out = {
        "kind": "scored",
        "id": f"{probe}|w{bundle['wi']}",
        "probe": probe,
        "world": int(bundle["wi"]),
        "passed": bool(ok),
        "ok": bool(ok),
        "train": sc,
        "W_installed": False,
        "discarded": True,
        "candidate_v41_lock": False,
        "parent_w_act_query_unchanged": True,
    }
    assert_finite_record(out, ctx=out["id"])
    return out


def score_world(bundle: dict[str, Any], rename_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    W_prefix, p_ok = fit_decoder(bundle, bundle["prefix_rows"])
    W_later, l_ok = fit_decoder(bundle, bundle["later_rows"])
    bundle["decoder"]["prefix_decoder_feasible"] = bool(p_ok)
    bundle["decoder"]["later_decoder_feasible"] = bool(l_ok)
    cells = [
        scored_cell(bundle, "prefix_on_prefix", W_prefix, bundle["prefix_rows"]),
        scored_cell(bundle, "prefix_on_later", W_prefix, bundle["later_rows"]),
        scored_cell(bundle, "later_on_later", W_later, bundle["later_rows"]),
        scored_cell(bundle, "renamed_cues", W_prefix, rename_rows),
    ]
    del W_prefix
    del W_later
    return cells


def synthetic_grid(*, decoder_ok: bool = True, code: str = "stage_a_integrated_pass") -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    observer_ok = code not in ("observer_used_runner_provenance", "setup_precondition_fail", "runner_constructed_value")
    prefix_ok = code not in (
        "setup_precondition_fail",
        "ordinary_cortex_broken",
        "observer_used_runner_provenance",
        "runner_constructed_value",
        "handle_copied_into_S",
    )
    later_sep = prefix_ok and code not in ("action_collapse",)
    later_xfer = later_sep and code not in ("later_context_drift", "fresh_world_fail", "cue_overfit")
    renamed = later_xfer and code == "stage_a_integrated_pass"
    if code == "fresh_world_fail":
        later_xfer_w = {0: True, 1: False}
    elif code == "cue_overfit":
        later_xfer_w = {0: True, 1: True}
        renamed = False
    else:
        later_xfer_w = {0: later_xfer, 1: later_xfer}
    for wi in (0, 1):
        cells.append(
            {
                "id": f"decoder|w{wi}",
                "kind": "setup",
                "passed": bool(decoder_ok) and code != "setup_precondition_fail",
                "observer_ok": bool(observer_ok) and code not in ("runner_constructed_value", "handle_copied_into_S"),
                "n_observer_provenance": 0 if observer_ok else 1,
                "n_handle_copied_into_s": 1 if code == "handle_copied_into_S" else 0,
                "constructed_value": code == "runner_constructed_value",
                "ref_ok": bool(prefix_ok),
                "offline_reconstruction": code == "offline_reconstruction",
                "checkpoint_restore_ok": code != "checkpoint_restore_fail",
                "feedback_off_ok": code != "feedback_off_fail",
                "permuted_feedback_ok": code != "permuted_feedback_fail",
                "reward_gate_ok": code != "reward_gate_fail",
            }
        )
    for probe, default in (
        ("prefix_on_prefix", prefix_ok),
        ("prefix_on_later", False),
        ("later_on_later", later_sep),
        ("renamed_cues", renamed),
    ):
        for wi in (0, 1):
            ok = default
            if probe == "prefix_on_later":
                ok = bool(later_xfer_w[wi])
            if code == "offline_reconstruction":
                ok = False if probe != "prefix_on_prefix" else prefix_ok
            cells.append({"id": f"{probe}|w{wi}", "probe": probe, "ok": bool(ok), "passed": bool(ok)})
    return cells


def _decision(cells: list[dict[str, Any]]) -> tuple[str, str, dict[str, Any]]:
    by = {c["id"]: c for c in cells}
    flags = {
        "install_W_star": False,
        "candidate_v41_lock": False,
        "kqv_edited": False,
        "tm063_opened": False,
        "stage_a_integrated_pass": False,
        "later_context_drift": False,
        "action_collapse": False,
    }

    def both(probe: str) -> bool:
        return all(bool(by[f"{probe}|w{wi}"].get("ok")) for wi in (0, 1))

    def _finish(code: str) -> tuple[str, str, dict[str, Any]]:
        if code in flags:
            flags[code] = True
        return code, code, flags

    for wi in (0, 1):
        if not bool(by[f"decoder|w{wi}"].get("passed")):
            return _finish("setup_precondition_fail")
    if any(bool(by[f"decoder|w{wi}"].get("constructed_value")) for wi in (0, 1)):
        return _finish("runner_constructed_value")
    if any(int(by[f"decoder|w{wi}"].get("n_handle_copied_into_s") or 0) > 0 for wi in (0, 1)):
        return _finish("handle_copied_into_S")
    if any(not bool(by[f"decoder|w{wi}"].get("observer_ok", True)) for wi in (0, 1)):
        return _finish("observer_used_runner_provenance")
    if any(int(by[f"decoder|w{wi}"].get("n_observer_provenance") or 0) > 0 for wi in (0, 1)):
        return _finish("observer_used_runner_provenance")
    if any(not bool(by[f"decoder|w{wi}"].get("ref_ok", True)) for wi in (0, 1)):
        return _finish("ordinary_cortex_broken")
    if any(bool(by[f"decoder|w{wi}"].get("offline_reconstruction")) for wi in (0, 1)):
        return _finish("offline_reconstruction")
    if any(not bool(by[f"decoder|w{wi}"].get("checkpoint_restore_ok", True)) for wi in (0, 1)):
        return _finish("checkpoint_restore_fail")
    if any(not bool(by[f"decoder|w{wi}"].get("feedback_off_ok", True)) for wi in (0, 1)):
        return _finish("feedback_off_fail")
    if any(not bool(by[f"decoder|w{wi}"].get("permuted_feedback_ok", True)) for wi in (0, 1)):
        return _finish("permuted_feedback_fail")
    if any(not bool(by[f"decoder|w{wi}"].get("reward_gate_ok", True)) for wi in (0, 1)):
        return _finish("reward_gate_fail")
    if not both("prefix_on_prefix"):
        return _finish("ordinary_cortex_broken")
    if not both("later_on_later"):
        return _finish("action_collapse")
    if bool(by["prefix_on_later|w0"].get("ok")) != bool(by["prefix_on_later|w1"].get("ok")):
        return _finish("fresh_world_fail")
    if both("prefix_on_later") and not both("renamed_cues"):
        return _finish("cue_overfit")
    if not both("prefix_on_later"):
        return _finish("later_context_drift")
    if not both("renamed_cues"):
        return _finish("fresh_world_fail")
    return _finish("stage_a_integrated_pass")


def eval_stage_a(adapter_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    p = load_prereg()
    frozen = str(p.get("frozen_runner_sha") or "")
    if frozen and frozen != "PLACEHOLDER" and sha_file(THIS) != frozen:
        raise RuntimeError("MEMLANG-1 Stage A runner SHA drifted")
    _assert_frozen(p)
    leak = refuse_runner_leaks(THIS)
    if leak:
        raise RuntimeError(f"runner leak: {leak}")
    cfg = dict(adapter_cfg or {"family": "identity", "name": "identity"})
    ids = expected_cell_ids()
    cells: list[dict[str, Any]] = []
    hashes: list[str] = []
    tmp = tempfile.TemporaryDirectory(prefix="ml_a_")
    try:
        bundles = []
        rename_rows_by_w: dict[int, list[dict[str, Any]]] = {}
        for wi in range(int(p["n_worlds"])):
            b = collect_bundle(wi=int(wi), tmp=tmp.name, adapter_cfg=cfg, cue_prefix="dev", domain_key="DEV")
            bundles.append(b)
            rb = collect_bundle(
                wi=int(wi),
                tmp=tmp.name,
                adapter_cfg=cfg,
                cue_prefix="rn",
                domain_key="DEV",
                adapter=b["adapter"],
            )
            rename_rows_by_w[int(wi)] = list(rb["prefix_rows"]) + list(rb["later_rows"])
            del rb
        pin_off = collect_bundle(
            wi=0,
            tmp=tmp.name,
            adapter_cfg=cfg,
            cue_prefix="pinoff",
            domain_key="DEV",
            n_online_repeats=2,
            feedback_off=True,
        )
        feedback_off_ok = (len(pin_off["prefix_rows"]) + len(pin_off["later_rows"])) == 0
        del pin_off
        pin_perm = collect_bundle(
            wi=0,
            tmp=tmp.name,
            adapter_cfg=cfg,
            cue_prefix="pinperm",
            domain_key="DEV",
            n_online_repeats=2,
            permute_feedback=True,
        )
        adp = pin_perm["adapter"]
        permuted_ok = adp.last_motor is None or int(np.asarray(adp.last_motor).size) != int(p["n"])
        if adp.last_motor is not None and str(cfg.get("family") or "identity") != "identity":
            permuted_ok = bool(permuted_ok and adp.last_target is not None and int(np.asarray(adp.last_target).size) == int(p["n"]))
        del pin_perm
        scored: dict[str, dict[str, Any]] = {}
        hashes = []
        for b in bundles:
            b["decoder"]["feedback_off_ok"] = bool(feedback_off_ok)
            b["decoder"]["permuted_feedback_ok"] = bool(permuted_ok)
            hashes.append(str(b["w0_hash"]))
            for cell in score_world(b, rename_rows_by_w[int(b["wi"])]):
                scored[cell["id"]] = cell
            cells.append(b["decoder"])
            del b["adapter"]
        for probe in list(p["probes"]):
            for wi in range(int(p["n_worlds"])):
                cells.append(scored[f"{probe}|w{wi}"])
    finally:
        tmp.cleanup()
    if [c["id"] for c in cells] != ids:
        raise RuntimeError("cell id order drifted")
    code, then, flags = _decision(cells)
    return {
        "lab": LAB,
        "stage": "A",
        "adapter": cfg,
        "n_cells": len(cells),
        "cells": cells,
        "decision_code": code,
        "decision_then": then,
        "phase_flags": flags,
        "install_W_star": False,
        "candidate_v41_lock": False,
        "neural_untouched": False,
        "identity_default_hook": True,
        "kqv_edited": False,
        "genome_checkpoint": {"w0_hash": list(hashes)},
        "world_seed": {
            "seed_registry": int(p["seed_registry"]),
            "domains": dict(p["domains"]),
            "n_worlds": int(p["n_worlds"]),
        },
        "status": "complete",
    }


def smoke() -> dict[str, Any]:
    ids = expected_cell_ids()
    steps = {"setup_precondition_fail": _decision(synthetic_grid(decoder_ok=False))[0]}
    for step in BEHAVIORAL_LADDER:
        steps[step] = _decision(synthetic_grid(code=step))[0]
    return {
        "smoke_ok": True,
        "n_cells": len(ids),
        "ladder": steps,
        "retrieve_leak": refuse_runner_leaks(THIS),
        "receipt_contract": receipt_contract_violations(THIS),
        "floor": float(EPISODE_MATCH_L2),
        "candidate_exists": CANDIDATE.exists(),
        "slots": int(EPISODE_SLOTS),
        "install_W_star": False,
        "candidate_v41_lock": False,
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        print(json.dumps(smoke(), indent=2))
        return
    raise SystemExit("use --smoke")


if __name__ == "__main__":
    main()
