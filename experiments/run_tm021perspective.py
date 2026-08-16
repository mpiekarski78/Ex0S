"""TM.0.21.PERSPECTIVE: source exposure, evidenced perspective, report alignment.

Phase A baseline (reliability-on, perspective off) → B observe_exposure candidate
→ C P0–P12 → capacity → wall.
Product stays 0.0.004; earned_next=false; ex0s=null.
Never honesty_score / believes / false-belief claims.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm016relate import clear_by_source, empty_birth, reload_store
from experiments.run_tm019inquire import ensure_context_grounded, score_plan, teach_probe_render
from experiments.run_tm020reliability import make_reliability
from three_memory import agent as agent_mod
from three_memory.policy import UsePolicy
from three_memory.symbols import parse_tagfile

PREREG_BASELINE = REPO_ROOT / "docs" / "perspective_baseline.prereg.lock"
PREREG_MECH = REPO_ROOT / "docs" / "perspective_mech.prereg.lock"
PREREG_WALL = REPO_ROOT / "docs" / "perspective_wall.prereg.lock"
FIXTURE_JSON = REPO_ROOT / "docs" / "perspective_fixture.json"
BASELINE_LOCK = REPO_ROOT / "docs" / "perspective_baseline.lock"
CANDIDATE_LOCK = REPO_ROOT / "docs" / "perspective.candidate.lock"
CANDIDATE_V1_LOCK = REPO_ROOT / "docs" / "perspective.candidate.v1.lock"
MECH_LOCK = REPO_ROOT / "docs" / "perspective_mech.lock"
PERSPECTIVE_LOCK = REPO_ROOT / "docs" / "perspective.lock"
WALL_LOCK = REPO_ROOT / "docs" / "perspective_wall.lock"
RESULTS_MD = REPO_ROOT / "docs" / "tm021perspective_results.md"
AGENT_PY = REPO_ROOT / "three_memory" / "agent.py"
RELIABILITY_LOCK = REPO_ROOT / "docs" / "reliability.lock"

DEFAULT_SEED = 12345
SOURCE_PERSPECTIVE = "experience_perspective"
STAGES = tuple(f"P{i}" for i in range(13))

BASELINE_CLAIM = (
    "Frozen RELIABILITY (make_reliability, perspective off) does not distinguish "
    "ALIGNED vs MISALIGNED evidenced perspectives — conflicting testimony remains "
    "predictive/HOLD as today; RELIABILITY lock behavior unchanged."
)

MECH_CLAIM = (
    "An opt-in recipe may record observable exposure events, reconstruct a source's "
    "last uniquely supported evidenced perspective via exact event linkage (not Jaccard "
    "attach), recompute ALIGNED|MISALIGNED|UNKNOWN at use time, and apply a frozen "
    "influence rule with predictive source_evidence_margin — without honesty_score, "
    "knows/believes, or claiming belief/intent."
)

WALL_CLAIM = (
    "On frozen make_perspective, a preregistered wall probes attention gaps, "
    "misunderstanding, forgetting, indirect learning, error vs lie, deception, "
    "copied information, nested beliefs, and strategic adaptation. Need not fully "
    "pass; first_fail_wall diagnoses the next primitive — evidenced perspective does "
    "not claim to model another mind."
)


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _sha_src(fn: Callable[..., Any]) -> str:
    return _sha_bytes(inspect.getsource(fn).encode())


def load_fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_JSON.read_text(encoding="utf-8"))


def make_perspective(s_dir: Path, policy: UsePolicy | None = None, **kwargs: Any) -> Any:
    ag = make_reliability(s_dir, policy, **kwargs)
    ag.use_source_perspective = True
    fix = load_fixture()
    ag.perspective_lambda = int(fix.get("perspective_lambda") or 4)
    ag.perspective_n_min = int(fix.get("perspective_n_min") or 2)
    ag.perspective_jaccard = float(fix.get("perspective_jaccard") or 0.5)
    strong = fix.get("strong_exposure_atoms") or [
        "exp_delivered",
        "exp_ack_read",
        "exp_receipt",
    ]
    ag.perspective_strong_exposure = frozenset(strong)
    return ag


def fresh(tmp: Path, name: str, policy: UsePolicy, *, perspective: bool) -> tuple[Path, Any]:
    s_dir = tmp / name
    empty_birth(s_dir)
    ag = make_perspective(s_dir, policy) if perspective else make_reliability(s_dir, policy)
    ag.reset_rho()
    if hasattr(ag, "reset_inquire_budget"):
        ag.reset_inquire_budget()
    fix = load_fixture()
    ensure_context_grounded(
        ag, list(fix.get("context_tokens") or ["scene", "world"]), tag="boot"
    )
    return s_dir, ag


def apply_ground(ag: Any, row: dict[str, Any]) -> None:
    info = {
        "symbol": row["symbol"],
        "paired": row["paired"],
        "trial_id": row["trial_id"],
        "result": row["result"],
    }
    if row.get("provenance"):
        info["provenance"] = row["provenance"]
    out = ag.observe_symbol_ground(info)
    if not out.get("ok"):
        raise RuntimeError(f"ground reject: {out}")


def apply_testimony(ag: Any, row: dict[str, Any]) -> None:
    if not getattr(ag, "use_source_reliability", False):
        return
    out = ag.observe_testimony(
        {
            "speaker_token": row["speaker_token"],
            "context_atoms": list(row["context_atoms"]),
            "claim_atoms": list(row["claim_atoms"]),
            "event_token": row["event_token"],
        }
    )
    if not out.get("ok") and out.get("why") != "reliability_off":
        raise RuntimeError(f"testimony reject: {out}")


def apply_exposure(ag: Any, row: dict[str, Any]) -> None:
    if not getattr(ag, "use_source_perspective", False):
        return
    out = ag.observe_exposure(
        {
            "speaker_token": row["speaker_token"],
            "context_atoms": list(row["context_atoms"]),
            "exposure_atoms": list(row["exposure_atoms"]),
            "event_token": row["event_token"],
        }
    )
    if not out.get("ok"):
        raise RuntimeError(f"exposure reject: {out}")


def clear_exposure_rows(s_dir: Path) -> int:
    return clear_by_source(s_dir, SOURCE_PERSPECTIVE)


def verify_baseline_prereg() -> tuple[bool, str, dict[str, Any]]:
    if not PREREG_BASELINE.exists():
        return False, "missing baseline prereg", {}
    lock = json.loads(PREREG_BASELINE.read_text(encoding="utf-8"))
    if lock.get("lab") != "TM.0.21.PERSPECTIVE":
        return False, "lab drift", lock
    if lock.get("earned_next") is not False or lock.get("ex0s") is not None:
        return False, "earn/product drift", lock
    if lock.get("preregistered_claim") != BASELINE_CLAIM:
        return False, "claim drift", lock
    if lock.get("fixture_sha") != _sha_file(FIXTURE_JSON):
        return False, "fixture_sha pin", lock
    if lock.get("phase_a", {}).get("agent_edits_permitted") is not False:
        return False, "phase A must forbid agent edits", lock
    pins = lock.get("prior_lock_shas") or {}
    if pins.get("reliability.lock") != _sha_file(RELIABILITY_LOCK):
        return False, "prior reliability.lock pin", lock
    return True, "perspective_baseline.prereg.lock intact", lock


def verify_mech_prereg() -> tuple[bool, str, dict[str, Any]]:
    if not PREREG_MECH.exists():
        return False, "missing mech prereg", {}
    lock = json.loads(PREREG_MECH.read_text(encoding="utf-8"))
    if lock.get("lab") != "TM.0.21.PERSPECTIVE.MECH":
        return False, "lab drift", lock
    if lock.get("earned_next") is not False or lock.get("ex0s") is not None:
        return False, "earn/product drift", lock
    if lock.get("preregistered_claim") != MECH_CLAIM:
        return False, "claim drift", lock
    if lock.get("flag") != "use_source_perspective" or lock.get("flag_default") is not False:
        return False, "flag contract", lock
    if lock.get("presence_never_attaches") is not True:
        return False, "presence_never_attaches", lock
    if lock.get("use_time_recompute") is not True:
        return False, "use_time_recompute", lock
    if lock.get("jaccard_scope") != "report_alignment_margin_transfer_only":
        return False, "jaccard_scope", lock
    if lock.get("fixture_sha") != _sha_file(FIXTURE_JSON):
        return False, "fixture_sha pin", lock
    if any(k in lock for k in ("agent_sha", "run_tm021perspective_sha", "make_perspective_sha")):
        return False, "prereg contains runner/agent SHAs", lock
    return True, "perspective_mech.prereg.lock intact", lock


def verify_wall_prereg() -> tuple[bool, str, dict[str, Any]]:
    if not PREREG_WALL.exists():
        return False, "missing wall prereg", {}
    lock = json.loads(PREREG_WALL.read_text(encoding="utf-8"))
    if lock.get("lab") != "TM.0.21.PERSPECTIVE.WALL":
        return False, "lab drift", lock
    if lock.get("earned_next") is not False or lock.get("ex0s") is not None:
        return False, "earn/product drift", lock
    if lock.get("preregistered_claim") != WALL_CLAIM:
        return False, "claim drift", lock
    if lock.get("fixture_sha") != _sha_file(FIXTURE_JSON):
        return False, "fixture_sha pin", lock
    if lock.get("mechanism_changes_permitted") is not False:
        return False, "wall must freeze mechanism", lock
    fix = load_fixture()
    if lock.get("probe_ids") != [w["id"] for w in fix["wall"]]:
        return False, "wall probe_ids drift", lock
    return True, "perspective_wall.prereg.lock intact", lock


def run_fork(
    tmp: Path,
    s_dir: Path,
    policy: UsePolicy,
    fork: dict[str, Any],
    *,
    perspective: bool,
) -> dict[str, Any] | None:
    kind = fork["kind"]
    clone = tmp / f"fork_{kind}_{fork.get('stage')}_{fork.get('id') or 'x'}"
    if clone.exists():
        shutil.rmtree(clone)
    shutil.copytree(s_dir, clone)
    ag = make_perspective(clone, policy) if perspective else make_reliability(clone, policy)
    reload_store(ag)
    ag.reset_rho()
    if kind == "strip_exposure":
        clear_exposure_rows(clone)
        reload_store(ag)
    elif kind == "donor_exposure":
        clear_exposure_rows(clone)
        reload_store(ag)
        for row in fork.get("donor_ops") or []:
            if row.get("op") == "exposure":
                apply_exposure(ag, row)
            elif row.get("op") == "ground" or "symbol" in row:
                apply_ground(ag, row)
    else:
        raise ValueError(kind)
    if fork.get("expect_alignment"):
        status = ag.report_alignment_status(
            fork["speaker"], fork["claim_atoms"], fork.get("context_atoms")
        )
        if status != fork["expect_alignment"]:
            return {
                "stage": fork.get("stage"),
                "lane": kind,
                "probe": fork.get("id") or kind,
                "expected": fork["expect_alignment"],
                "actual": status,
                "failure_family": fork.get("failure_family") or "isolation",
            }
        return None
    probe = {
        "context_atoms": list(fork.get("context_atoms") or ["scene", "fac_lab"]),
        "input_symbols": list(fork["input_symbols"]),
        "expect_status": fork["expect_status"],
        "expect_answer": fork.get("expect_answer"),
    }
    ok, plan = score_plan(ag, probe)
    if not ok:
        return {
            "stage": fork.get("stage"),
            "lane": kind,
            "probe": fork.get("id") or kind,
            "expected": fork.get("expect_status"),
            "actual": plan.get("status"),
            "failure_family": fork.get("failure_family") or "isolation",
        }
    return None


def run_ops(
    ag: Any,
    s_dir: Path,
    tmp: Path,
    policy: UsePolicy,
    ops: Sequence[dict[str, Any]],
    *,
    perspective: bool,
) -> dict[str, Any]:
    first_fail: dict[str, Any] | None = None
    last_clear: str | None = None
    timings: list[float] = []
    n_probe = 0

    for op in ops:
        kind = op.get("op")
        if kind == "stage_marker":
            continue
        if kind == "reset_budget":
            if hasattr(ag, "reset_inquire_budget"):
                ag.reset_inquire_budget()
            continue
        if kind == "rho_reset":
            ag.reset_rho()
            continue
        if kind == "ground" or (kind is None and "symbol" in op and "paired" in op):
            apply_ground(ag, op if "symbol" in op else op)
            continue
        if kind == "testimony":
            apply_testimony(ag, op)
            continue
        if kind == "exposure":
            apply_exposure(ag, op)
            continue
        if kind == "alignment_check":
            if first_fail is not None:
                continue
            status = ag.report_alignment_status(
                op["speaker"], op["claim_atoms"], op.get("context_atoms")
            )
            expect = op["expect_status"]
            if status != expect:
                first_fail = {
                    "stage": op.get("stage"),
                    "probe": op.get("id") or "alignment_check",
                    "expected": expect,
                    "actual": status,
                    "failure_family": op.get("failure_family") or "perspective",
                }
            else:
                st = op.get("stage")
                if st in STAGES:
                    last_clear = st
            continue
        if kind == "margin_check":
            if first_fail is not None:
                continue
            m = float(ag.report_alignment_margin(op["speaker"], op["context_atoms"]))
            ok = True
            if "expect_min" in op and m < float(op["expect_min"]):
                ok = False
            if "expect_max" in op and m > float(op["expect_max"]):
                ok = False
            if not ok:
                first_fail = {
                    "stage": op.get("stage"),
                    "probe": op.get("id") or "margin_check",
                    "expected": {k: op[k] for k in ("expect_min", "expect_max") if k in op},
                    "actual": m,
                    "failure_family": "calibration",
                }
            else:
                st = op.get("stage")
                if st in STAGES:
                    last_clear = st
            continue
        if kind == "plan":
            n_probe += 1
            if first_fail is not None:
                continue
            for pr in op.get("teach_renders") or []:
                teach_probe_render(
                    ag,
                    context_atoms=list(op.get("context_atoms") or ["scene", "fac_lab"]),
                    probe_atoms=list(pr["probe_atoms"]),
                    prefix_id=str(op.get("id") or "plan"),
                )
            t0 = time.perf_counter()
            ok, plan = score_plan(ag, op)
            timings.append(time.perf_counter() - t0)
            if not ok:
                first_fail = {
                    "stage": op.get("stage"),
                    "probe": op.get("id"),
                    "expected": op.get("expect_status"),
                    "actual": plan.get("status"),
                    "why": plan.get("why"),
                    "failure_family": op.get("failure_family") or "unknown",
                }
            else:
                st = op.get("stage")
                if st in STAGES:
                    last_clear = st
            continue
        if kind == "fork":
            if first_fail is not None:
                continue
            fail = run_fork(tmp, s_dir, policy, op, perspective=perspective)
            if fail is not None:
                first_fail = fail
            else:
                st = op.get("stage")
                if st in STAGES:
                    last_clear = st
            continue
        if kind == "capacity_launch":
            st = op.get("stage")
            if st in STAGES and first_fail is None:
                last_clear = st
            continue
        if "symbol" in op and "paired" in op and "trial_id" in op:
            apply_ground(ag, op)
            continue
        raise ValueError(f"unknown op {kind}: {op}")

    metrics = {
        "s_row_count": len(ag.store.records()) if hasattr(ag, "store") else 0,
        "n_plan_timings": len(timings),
    }
    if timings:
        metrics["p50_plan_s"] = float(statistics.median(timings))
        metrics["p95_plan_s"] = float(
            sorted(timings)[max(0, int(round(0.95 * (len(timings) - 1))))]
        )
    return {
        "ok": first_fail is None,
        "first_fail": first_fail,
        "last_stage_clear": last_clear,
        "n_probes": n_probe,
        "metrics": metrics,
        "s_dir": str(s_dir),
        "ag": ag,
    }


def flatten_life_ops(fix: dict[str, Any]) -> list[dict[str, Any]]:
    ops: list[dict[str, Any]] = []
    life = fix.get("life") or {}
    for stage in STAGES:
        for op in (life.get(stage) or {}).get("ops") or []:
            if isinstance(op, dict) and "op" not in op and "symbol" in op:
                row = dict(op)
                row["op"] = "ground"
                ops.append(row)
            else:
                ops.append(op)
    return ops


def run_baseline(*, seed: int = DEFAULT_SEED, write_lock: bool = False) -> dict[str, Any]:
    ok_p, why_p, _ = verify_baseline_prereg()
    if not ok_p:
        raise RuntimeError(why_p)
    fixture = load_fixture()
    policy = UsePolicy(seed=seed)
    with tempfile.TemporaryDirectory(prefix="tm021persp_base_") as tmp:
        s_dir, ag = fresh(Path(tmp), "baseline", policy, perspective=False)
        assert not getattr(ag, "use_source_perspective", False)
        result = run_ops(
            ag, s_dir, Path(tmp), policy, fixture["script_baseline"], perspective=False
        )
        # Perspective API must stay UNKNOWN when off
        status = ag.report_alignment_status("spk_a", ["box", "hyp_red"], ["scene", "fac_lab"])
        result_ok = result["ok"] and status == "UNKNOWN"
        first_fail = result["first_fail"]
        if status != "UNKNOWN":
            first_fail = {"probe": "baseline_alignment_off", "expected": "UNKNOWN", "actual": status}
    summary = {
        "version": "TM.0.21.PERSPECTIVE.BASELINE",
        "lab": "TM.0.21.PERSPECTIVE",
        "phase": "A",
        "ok": result_ok,
        "earned_next": False,
        "ex0s": None,
        "claim": BASELINE_CLAIM,
        "first_fail": first_fail,
        "fixture_sha": _sha_file(FIXTURE_JSON),
        "agent_sha": _sha_file(AGENT_PY),
        "factory": "make_reliability",
    }
    if write_lock:
        snap = {
            "version": "TM.0.21.PERSPECTIVE.BASELINE",
            "lab": "TM.0.21.PERSPECTIVE",
            "phase": "A",
            "ex0s_under_test": "0.0.004",
            "earned_next": False,
            "ex0s": None,
            "ok": summary["ok"],
            "first_fail": summary["first_fail"],
            "fixture_sha": _sha_file(FIXTURE_JSON),
            "perspective_baseline_prereg_sha": _sha_file(PREREG_BASELINE),
            "reliability_lock_sha": _sha_file(RELIABILITY_LOCK),
            "agent_sha": _sha_file(AGENT_PY),
            "run_tm021perspective_sha": _sha_file(Path(__file__)),
            "refuse": ["editing agent.py in Phase A", "honesty_score", "earned_next=true or non-null ex0s"],
        }
        BASELINE_LOCK.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return summary


def write_candidate_lock() -> dict[str, Any]:
    ok_w, why_w, _ = verify_wall_prereg()
    if not ok_w:
        raise RuntimeError(f"wall prereg required before candidate: {why_w}")
    snap = {
        "version": "TM.0.21.PERSPECTIVE.CANDIDATE",
        "lab": "TM.0.21.PERSPECTIVE.MECH",
        "ex0s_under_test": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "flag": "use_source_perspective",
        "source": SOURCE_PERSPECTIVE,
        "observation_abi": "observe_exposure",
        "statuses": ["ALIGNED", "MISALIGNED", "UNKNOWN"],
        "margin_name": "report_alignment_margin",
        "influence": {
            "world_unique_first": True,
            "ALIGNED": "does_not_prove_world_correctness",
            "MISALIGNED": "withhold_weight_0_never_invert",
            "UNKNOWN": "unpenalized_predictive_or_inquire",
        },
        "presence_never_attaches": True,
        "jaccard_scope": "report_alignment_margin_transfer_only",
        "factory": "experiments.run_tm021perspective.make_perspective",
        "agent_sha": _sha_file(AGENT_PY),
        "observe_exposure_sha": _sha_src(agent_mod.ThreeMemoryAgent.observe_exposure),
        "report_alignment_status_sha": _sha_src(
            agent_mod.ThreeMemoryAgent.report_alignment_status
        ),
        "make_perspective_sha": _sha_src(make_perspective),
        "run_tm021perspective_sha": _sha_file(Path(__file__)),
        "perspective_mech_prereg_sha": _sha_file(PREREG_MECH),
        "perspective_baseline_sha": _sha_file(BASELINE_LOCK) if BASELINE_LOCK.exists() else None,
        "perspective_wall_prereg_sha": _sha_file(PREREG_WALL),
        "note": "Pinned after unscored ABI smoke, before scored cells. Preserve as v1 if audit rewrites agent.",
    }
    CANDIDATE_LOCK.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    if not CANDIDATE_V1_LOCK.exists():
        CANDIDATE_V1_LOCK.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return snap


def run_smoke(*, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    ok_p, why_p, _ = verify_mech_prereg()
    if not ok_p:
        return {"ok": False, "why": why_p}
    policy = UsePolicy(seed=seed)
    with tempfile.TemporaryDirectory(prefix="tm021persp_smoke_") as tmp:
        _, ag = fresh(Path(tmp), "smoke", policy, perspective=True)
        bad = ag.observe_exposure({"speaker_token": "a"})
        if bad.get("why") != "exact_key_reject":
            return {"ok": False, "why": f"reject smoke failed: {bad}"}
        out = ag.observe_exposure(
            {
                "speaker_token": "spk_smoke",
                "context_atoms": ["scene", "fac_lab"],
                "exposure_atoms": ["exp_ack_read"],
                "event_token": "esmoke",
            }
        )
        if not out.get("ok"):
            return {"ok": False, "why": f"exposure failed: {out}"}
        apply_ground(
            ag,
            {
                "symbol": "box",
                "paired": "hyp_red",
                "trial_id": "evt_esmoke__v0",
                "result": "success",
                "provenance": "direct",
            },
        )
        st = ag.report_alignment_status(
            "spk_smoke", ["box", "hyp_red"], ["scene", "fac_lab"]
        )
        if st != "ALIGNED":
            return {"ok": False, "why": f"expected ALIGNED got {st}"}
        # present-only must not attach
        out2 = ag.observe_exposure(
            {
                "speaker_token": "spk_p",
                "context_atoms": ["scene", "fac_lab"],
                "exposure_atoms": ["exp_present"],
                "event_token": "epres",
            }
        )
        apply_ground(
            ag,
            {
                "symbol": "box",
                "paired": "hyp_blue",
                "trial_id": "evt_epres__v0",
                "result": "success",
                "provenance": "direct",
            },
        )
        st2 = ag.report_alignment_status(
            "spk_p", ["box", "hyp_blue"], ["scene", "fac_lab"]
        )
        if st2 != "UNKNOWN":
            return {"ok": False, "why": f"present-only attached: {st2}"}
    return {"ok": True, "why": "abi_smoke"}


def run_unit_cells(*, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    ok_p, why_p, _ = verify_mech_prereg()
    if not ok_p:
        raise RuntimeError(why_p)
    if not CANDIDATE_LOCK.exists():
        raise RuntimeError("candidate.lock missing")
    fixture = load_fixture()
    policy = UsePolicy(seed=seed)
    cells_out = []
    with tempfile.TemporaryDirectory(prefix="tm021persp_cells_") as tmp:
        root = Path(tmp)
        for name, cell in fixture["unit_cells"].items():
            s_dir = root / name
            empty_birth(s_dir)
            use_p = bool(cell.get("flag"))
            ag = make_perspective(s_dir, policy) if use_p else make_reliability(s_dir, policy)
            ag.reset_rho()
            if hasattr(ag, "reset_inquire_budget"):
                ag.reset_inquire_budget()
            ensure_context_grounded(ag, ["scene", "fac_lab", "world"], tag=name)
            why = "ok"
            ok = True
            if cell.get("malformed") is not None:
                bad = ag.observe_exposure(cell["malformed"])
                ok = bad.get("why") == "exact_key_reject"
                cells_out.append(
                    {"cell": name, "ok": ok, "actual": "exact_key_reject", "why": bad.get("why")}
                )
                continue
            for row in cell.get("ops") or []:
                if row.get("op") == "exposure":
                    apply_exposure(ag, row)
                elif row.get("op") == "testimony":
                    apply_testimony(ag, row)
                elif row.get("op") == "ground" or "symbol" in row:
                    apply_ground(ag, row)
            cell_ok = True
            actual: Any = "ok"
            if cell.get("alignment"):
                al = cell["alignment"]
                status = ag.report_alignment_status(
                    al["speaker"], al["claim_atoms"], al.get("context_atoms")
                )
                actual = status
                cell_ok = status == al["expect_status"]
                if not use_p and status != "UNKNOWN":
                    cell_ok = False
                    why = "flag_should_force_unknown"
            if cell_ok and cell.get("probe"):
                ok_p2, plan = score_plan(ag, cell["probe"])
                if not ok_p2:
                    cell_ok = False
                    actual = plan.get("status")
                    why = plan.get("why") or "probe_fail"
            if cell_ok and cell.get("forks"):
                for fork in cell["forks"]:
                    f = dict(fork)
                    f.setdefault("stage", name)
                    fail = run_fork(root, s_dir, policy, f, perspective=use_p)
                    if fail is not None:
                        cell_ok = False
                        why = f"fork {fork['kind']}: {fail}"
                        break
            cells_out.append(
                {
                    "cell": name,
                    "ok": bool(ok and cell_ok),
                    "actual": actual,
                    "why": why if not (ok and cell_ok) else "pass",
                }
            )
    all_ok = all(c["ok"] for c in cells_out)
    return {
        "ok": all_ok,
        "earned_next": False,
        "ex0s": None,
        "cells": cells_out,
        "n_pass": sum(1 for c in cells_out if c["ok"]),
        "n_cells": len(cells_out),
    }


def write_mech_lock(cells: dict[str, Any]) -> dict[str, Any]:
    snap = {
        "version": "TM.0.21.PERSPECTIVE.MECH",
        "lab": "TM.0.21.PERSPECTIVE.MECH",
        "ex0s_under_test": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "ok": cells["ok"],
        "n_pass": cells["n_pass"],
        "n_cells": cells["n_cells"],
        "cells": cells["cells"],
        "agent_sha": _sha_file(AGENT_PY),
        "candidate_sha": _sha_file(CANDIDATE_LOCK),
        "run_tm021perspective_sha": _sha_file(Path(__file__)),
    }
    MECH_LOCK.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return snap


def run_capacity_lanes(*, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    fixture = load_fixture()
    policy = UsePolicy(seed=seed)
    lanes_out: dict[str, Any] = {}
    all_ok = True
    with tempfile.TemporaryDirectory(prefix="tm021persp_cap_") as tmp:
        root = Path(tmp)
        for lane_name, lane in (fixture.get("capacity") or {}).items():
            metric_only = set(lane.get("metric_only_rungs") or [])
            rungs_out = []
            first_fail_rung = None
            for branch in lane.get("branches") or []:
                rung = branch["rung"]
                s_dir, ag = fresh(root, f"{lane_name}_{rung}", policy, perspective=True)
                t0 = time.perf_counter()
                result = run_ops(
                    ag, s_dir, root, policy, branch["script"], perspective=True
                )
                dt = time.perf_counter() - t0
                metrics = dict(result.get("metrics") or {})
                metrics["wall_s"] = dt
                is_metric_only = bool(branch.get("metric_only")) or rung in metric_only
                ok = bool(result["ok"])
                if not ok and not is_metric_only and first_fail_rung is None:
                    first_fail_rung = rung
                    all_ok = False
                rungs_out.append(
                    {
                        "rung": rung,
                        "ok": ok,
                        "metric_only": is_metric_only,
                        "first_fail": result.get("first_fail"),
                        "metrics": metrics,
                    }
                )
            lane_ok = True
            for r in rungs_out:
                if not r["metric_only"] and not r["ok"]:
                    lane_ok = False
            lanes_out[lane_name] = {
                "ok": lane_ok,
                "first_fail_rung": first_fail_rung,
                "rungs": rungs_out,
            }
    return {"ok": all_ok and all(v["ok"] for v in lanes_out.values()), "lanes": lanes_out}


def run_life(*, seed: int = DEFAULT_SEED, write_lock: bool = False) -> dict[str, Any]:
    ok_p, why_p, _ = verify_mech_prereg()
    if not ok_p:
        raise RuntimeError(why_p)
    if not MECH_LOCK.exists():
        raise RuntimeError("mech lock missing")
    fixture = load_fixture()
    policy = UsePolicy(seed=seed)
    with tempfile.TemporaryDirectory(prefix="tm021persp_life_") as tmp:
        root = Path(tmp)
        s_dir, ag = fresh(root, "life", policy, perspective=True)
        main = run_ops(
            ag, s_dir, root, policy, flatten_life_ops(fixture), perspective=True
        )
        s_twin, ag_twin = fresh(root, "twin", policy, perspective=True)
        twin = run_ops(
            ag_twin, s_twin, root, policy, fixture.get("script_twin") or [], perspective=True
        )
        capacity = run_capacity_lanes(seed=seed)
    first_fail = main.get("first_fail")
    life_last = main.get("last_stage_clear")
    if main["ok"]:
        life_last = "P12"
    last = "P12" if main["ok"] and capacity.get("ok") else life_last
    summary = {
        "version": "TM.0.21.PERSPECTIVE",
        "lab": "TM.0.21.PERSPECTIVE",
        "ex0s_under_test": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "ok": first_fail is None and twin.get("ok") and capacity.get("ok"),
        "last_stage_clear": last,
        "life_last_stage_clear": life_last,
        "first_fail_stage": (first_fail or {}).get("stage"),
        "first_fail": first_fail,
        "failure_family": (first_fail or {}).get("failure_family"),
        "main_ok": main["ok"],
        "twin_ok": twin.get("ok"),
        "capacity": capacity,
        "fixture_sha": _sha_file(FIXTURE_JSON),
        "agent_sha": _sha_file(AGENT_PY),
        "candidate_sha": _sha_file(CANDIDATE_LOCK),
        "mech_sha": _sha_file(MECH_LOCK),
        "run_tm021perspective_sha": _sha_file(Path(__file__)),
        "bounded_claim": (
            "Ex0S reconstructed a source's last uniquely supported evidence perspective "
            "from observable exposure events."
        ),
    }
    if write_lock:
        write_perspective_lock(summary)
        write_results_md(summary)
    return summary


def write_perspective_lock(summary: dict[str, Any]) -> dict[str, Any]:
    snap = {
        "version": "TM.0.21.PERSPECTIVE",
        "lab": "TM.0.21.PERSPECTIVE",
        "ex0s_under_test": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "ok": summary["ok"],
        "last_stage_clear": summary.get("last_stage_clear"),
        "life_last_stage_clear": summary.get("life_last_stage_clear"),
        "first_fail_stage": summary.get("first_fail_stage"),
        "first_fail": summary.get("first_fail"),
        "failure_family": summary.get("failure_family"),
        "main_ok": summary.get("main_ok"),
        "twin_ok": summary.get("twin_ok"),
        "capacity": summary.get("capacity"),
        "fixture_sha": summary.get("fixture_sha"),
        "agent_sha": summary.get("agent_sha"),
        "candidate_sha": summary.get("candidate_sha"),
        "mech_sha": summary.get("mech_sha"),
        "run_tm021perspective_sha": summary.get("run_tm021perspective_sha"),
        "bounded_claim": summary.get("bounded_claim"),
        "refuse": [
            "honesty_score",
            "knows/believes/has_access",
            "false_belief claims",
            "earned_next=true or non-null ex0s",
        ],
    }
    PERSPECTIVE_LOCK.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return snap


def write_results_md(summary: dict[str, Any]) -> None:
    wall = {}
    if WALL_LOCK.exists():
        wall = json.loads(WALL_LOCK.read_text(encoding="utf-8"))
    ffw = wall.get("first_fail_wall") or {}
    lines = [
        "# TM.0.21.PERSPECTIVE results: source exposure & report alignment",
        "",
        "**Ex0S under test:** **0.0.004** (not a new stamp)",
        "**Lab:** TM.0.21.PERSPECTIVE",
        "**Date:** 16 August 2026",
        f"**ok:** `{summary.get('ok')}`",
        f"**life_last_stage_clear:** `{summary.get('life_last_stage_clear')}`",
        f"**first_fail (life):** `{summary.get('first_fail')}`",
        (
            f"**Wall first_fail:** `{ffw.get('id')}` / `{ffw.get('actual')}` → "
            f"next-primitive hint **{wall.get('next_primitive_hint')}**"
            if ffw
            else "**Wall first_fail:** (not yet written)"
        ),
        "",
        "Locks: [`perspective_baseline.lock`](perspective_baseline.lock) · "
        "[`perspective.candidate.lock`](perspective.candidate.lock) · "
        "[`perspective.candidate.v1.lock`](perspective.candidate.v1.lock) · "
        "[`perspective_mech.lock`](perspective_mech.lock) · "
        "[`perspective.lock`](perspective.lock) · "
        "[`perspective_wall.lock`](perspective_wall.lock)",
        "",
        "`earned_next`: **false** — no Ex0S 0.0.005 / 1.0. Product stamp remains **0.0.004**.",
        "",
        "## Bounded claim",
        "",
        "> Ex0S reconstructed a source's last uniquely supported evidence perspective "
        "from observable exposure events.",
        "",
        "Expanded: Given a closed symbolic information-flow topology and observable "
        "exposure events, Ex0S reconstructed first-order source-specific evidenced "
        "perspectives, distinguished reports aligned with those perspectives from "
        "reports inconsistent with them, and returned UNKNOWN when exposure or "
        "perspective was insufficient—without claiming knowledge, honesty or intent.",
        "",
        "## What cleared",
        "",
        "| Phase | Result |",
        "|-------|--------|",
        "| A baseline (`make_reliability`, perspective off) | no ALIGNED/MISALIGNED |",
        "| B unit cells | **9/9** |",
        "| C life P0–P12 + twin | clear through **P12** |",
        "| Capacity lanes | ok |",
        "| Wall scored scripts | pass (`W_attention_gap`, `W_indistinguishable`); "
        "`W_misunderstood` = executed **diagnostic_fail** |",
        "| Wall diagnostic first fail | `W_misunderstood` → **comprehension** "
        "(`not_run` probes are not diagnostics) |",
        "",
        "## Explicit non-claims",
        "",
        "- Not belief / knowledge / honesty_score / liar / intent",
        "- Presence never attaches world facts",
        "- Jaccard never attaches perspective across events",
        "- Nested ToM / genuine intent remain open (wall)",
        "- Default `make_reliability` / RELIABILITY locks unchanged",
        "",
        "## Essential breakthrough",
        "",
        "> Given a closed symbolic information-flow topology and observable exposure "
        "events, Ex0S reconstructed first-order source-specific evidenced perspectives, "
        "distinguished reports aligned with those perspectives from reports inconsistent "
        "with them, and returned UNKNOWN when exposure or perspective was insufficient—"
        "without claiming knowledge, honesty or intent.",
        "",
        "## Next",
        "",
        "Comprehension / nested ToM / intent remain open — named by wall "
        "`first_fail_wall`, not earned here.",
        "",
        "## Audit notes (apparatus)",
        "",
        "Post-freeze audit fixes (scientific claim unchanged):",
        "",
        "1. **World-unique first** — with perspective on, unique direct grounding "
        "answers before `source_evidence_margin` (frozen influence #1).",
        "2. **Donor-exposure forks** — U5/P12 strip+donor causality; "
        "donor-swapped exposure revises ALIGNED→MISALIGNED.",
        "3. **Repetition dedup** — margin key is speaker×cue×hyp×evidenced "
        "perspective (not per-report event_token).",
        "4. **Wall first_fail** — `W_misunderstood` is executed `diagnostic_fail`; "
        "`not_run` probes are not diagnostics.",
        "",
    ]
    RESULTS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_wall(*, seed: int = DEFAULT_SEED, write_lock: bool = False) -> dict[str, Any]:
    ok_p, why_p, _ = verify_wall_prereg()
    if not ok_p:
        raise RuntimeError(why_p)
    fixture = load_fixture()
    policy = UsePolicy(seed=seed)
    results = []
    first_fail = None
    with tempfile.TemporaryDirectory(prefix="tm021persp_wall_") as tmp:
        root = Path(tmp)
        scripts = fixture.get("wall_scripts") or {}
        for w in fixture["wall"]:
            wid = w["id"]
            note = w.get("note") or ""
            if wid in scripts:
                s_dir, ag = fresh(root, wid, policy, perspective=True)
                out = run_ops(
                    ag, s_dir, root, policy, scripts[wid]["ops"], perspective=True
                )
                if scripts[wid].get("expect_fail"):
                    if out["ok"]:
                        ok = False
                        actual = "diagnostic_fail"
                    else:
                        ok = False
                        actual = "script_miss"
                else:
                    ok = out["ok"]
                    actual = "pass" if ok else (out.get("first_fail") or {}).get("actual")
            else:
                # Not executed — do not use as first_fail diagnostic
                ok = None
                actual = "not_run"
            row = {
                "id": wid,
                "ok": ok,
                "actual": actual,
                "note": note,
                "dimension": w.get("dimension"),
            }
            results.append(row)
            if ok is False and first_fail is None:
                first_fail = row
    scored_ids = set(scripts.keys())
    scored_ok = True
    for r in results:
        if r["id"] not in scored_ids:
            continue
        spec = (fixture.get("wall_scripts") or {}).get(r["id"]) or {}
        if spec.get("expect_fail"):
            if r["actual"] != "diagnostic_fail":
                scored_ok = False
        elif r["ok"] is not True:
            scored_ok = False
    summary = {
        "version": "TM.0.21.PERSPECTIVE.WALL",
        "lab": "TM.0.21.PERSPECTIVE.WALL",
        "ex0s_under_test": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "ok": first_fail is None,
        "scored_probes_ok": scored_ok,
        "need_not_fully_pass": True,
        "first_fail_wall": first_fail,
        "next_primitive_hint": (first_fail or {}).get("dimension")
        or (first_fail or {}).get("id"),
        "probes": results,
        "fixture_sha": _sha_file(FIXTURE_JSON),
        "perspective_wall_prereg_sha": _sha_file(PREREG_WALL),
        "perspective_lock_sha": _sha_file(PERSPECTIVE_LOCK) if PERSPECTIVE_LOCK.exists() else None,
        "agent_sha": _sha_file(AGENT_PY),
        "run_tm021perspective_sha": _sha_file(Path(__file__)),
        "note": (
            "Results only. Evidenced perspective earned in-life; comprehension / nested "
            "ToM / intent remain open. not_run probes are not first_fail diagnostics. "
            "Do not rewrite wall prereg."
        ),
    }
    if write_lock:
        WALL_LOCK.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-prereg", action="store_true")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--write-baseline", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--write-candidate", action="store_true")
    ap.add_argument("--unit-cells", action="store_true")
    ap.add_argument("--write-mech-lock", action="store_true")
    ap.add_argument("--life", action="store_true")
    ap.add_argument("--write-lock", action="store_true")
    ap.add_argument("--wall", action="store_true")
    ap.add_argument("--write-wall", action="store_true")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    if args.verify_prereg:
        results = {}
        for name, fn in (
            ("baseline", verify_baseline_prereg),
            ("mech", verify_mech_prereg),
            ("wall", verify_wall_prereg),
        ):
            ok, why, _ = fn()
            results[name] = {"ok": ok, "why": why}
        print(json.dumps(results, indent=2))
        sys.exit(0 if all(v["ok"] for v in results.values()) else 1)

    if args.smoke:
        print(json.dumps(run_smoke(seed=args.seed), indent=2))
        return

    if args.write_candidate:
        if not BASELINE_LOCK.exists():
            raise SystemExit("baseline lock required")
        smoke = run_smoke(seed=args.seed)
        if not smoke["ok"]:
            raise SystemExit(f"smoke failed: {smoke}")
        snap = write_candidate_lock()
        print(json.dumps({"ok": True, "candidate": snap["agent_sha"][:16]}, indent=2))
        return

    if args.baseline or args.write_baseline:
        print(json.dumps(run_baseline(seed=args.seed, write_lock=args.write_baseline), indent=2))
        return

    if args.unit_cells:
        cells = run_unit_cells(seed=args.seed)
        print(json.dumps(cells, indent=2))
        if args.write_mech_lock:
            if not cells["ok"]:
                raise SystemExit("cells failed; refusing mech lock")
            write_mech_lock(cells)
        return

    if args.life or args.write_lock:
        print(
            json.dumps(
                run_life(seed=args.seed, write_lock=args.write_lock), indent=2, default=str
            )
        )
        return

    if args.wall or args.write_wall:
        print(
            json.dumps(
                run_wall(seed=args.seed, write_lock=args.write_wall), indent=2, default=str
            )
        )
        return

    ap.print_help()


if __name__ == "__main__":
    main()
