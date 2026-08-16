"""TM.0.20.RELIABILITY: context-conditioned predictive accuracy (source_evidence_margin).

Phase A baseline (inquire-on, reliability off) → B testimony/compare candidate → C R0–R12
→ unconfounded capacity → preregistered final wall.
Product stays 0.0.004; earned_next=false; ex0s=null.
Host never supplies confirm|contradict; organism compares claims to independent observes.
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
from experiments.run_tm019inquire import (
    ensure_context_grounded,
    make_inquire,
    score_plan,
    teach_probe_render,
)
from three_memory import agent as agent_mod
from three_memory.policy import UsePolicy
from three_memory.symbols import parse_tagfile

PREREG_BASELINE = REPO_ROOT / "docs" / "reliability_baseline.prereg.lock"
PREREG_MECH = REPO_ROOT / "docs" / "reliability_mech.prereg.lock"
PREREG_WALL = REPO_ROOT / "docs" / "reliability_wall.prereg.lock"
FIXTURE_JSON = REPO_ROOT / "docs" / "reliability_fixture.json"
BASELINE_LOCK = REPO_ROOT / "docs" / "reliability_baseline.lock"
CANDIDATE_LOCK = REPO_ROOT / "docs" / "reliability.candidate.lock"
CANDIDATE_V1_LOCK = REPO_ROOT / "docs" / "reliability.candidate.v1.lock"
MECH_LOCK = REPO_ROOT / "docs" / "reliability_mech.lock"
RELIABILITY_LOCK = REPO_ROOT / "docs" / "reliability.lock"
WALL_LOCK = REPO_ROOT / "docs" / "reliability_wall.lock"
RESULTS_MD = REPO_ROOT / "docs" / "tm020reliability_results.md"
CONTRACT_MD = REPO_ROOT / "docs" / "reliability_evidence_contract.md"
AGENT_PY = REPO_ROOT / "three_memory" / "agent.py"
INQUIRE_LOCK = REPO_ROOT / "docs" / "inquire.lock"

DEFAULT_SEED = 12345
SOURCE_TESTIMONY = "experience_testimony"
SOURCE_RELIABILITY = "experience_reliability"
SOURCE_GROUND = "experience_grounding"
SOURCE_INQUIRE = "experience_inquire"
STAGES = tuple(f"R{i}" for i in range(13))

BASELINE_CLAIM = (
    "Frozen INQUIRE (make_inquire, reliability off) still HOLDs on conflicting "
    "equal teachers with no spontaneous source weighting — INQUIRE lock behavior unchanged."
)

MECH_CLAIM = (
    "An opt-in recipe may record opaque-channel testimony, compare claim_atoms to "
    "independently observed world atoms (provenance ∈ {direct, experiment, state_read}), "
    "append experience_reliability rows, recompute bounded source_evidence_margin "
    "(λ=4, n_min=2), weight non-duplicated later conflicts, and keep inquiry on "
    "minimax value/cost — without host confirm|contradict, trust_score, or collapsing "
    "honesty/access/independence/intent into one score."
)

WALL_CLAIM = (
    "On frozen make_reliability, a preregistered wall probes the six social-cognition "
    "dimensions (competence/honesty/access/stability/independence/intent) plus "
    "indistinguishable-cause HOLD, aliases, unverifiable claims, and circular testimony. "
    "Need not fully pass; first_fail_wall diagnoses the next primitive — predictive "
    "accuracy does not claim to solve social understanding."
)


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _sha_src(fn: Callable[..., Any]) -> str:
    return _sha_bytes(inspect.getsource(fn).encode())


def load_fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_JSON.read_text(encoding="utf-8"))


def make_reliability(s_dir: Path, policy: UsePolicy | None = None, **kwargs: Any) -> Any:
    ag = make_inquire(s_dir, policy, **kwargs)
    ag.use_source_reliability = True
    ag.use_inquire_liveness = True
    fix = load_fixture()
    ag.reliability_lambda = int(fix.get("reliability_lambda") or 4)
    ag.reliability_n_min = int(fix.get("reliability_n_min") or 2)
    ag.reliability_jaccard = float(fix.get("reliability_jaccard") or 0.5)
    ag.inquire_budget = int(fix.get("inquire_budget") or 8)
    ag.inquire_cost_ask = int(fix.get("cost_ask") or 2)
    ag.inquire_cost_experiment = int(fix.get("cost_experiment") or 5)
    ag.reset_inquire_budget()
    return ag


def fresh(
    tmp: Path, name: str, policy: UsePolicy, *, reliability: bool
) -> tuple[Path, Any]:
    s_dir = tmp / name
    empty_birth(s_dir)
    ag = make_reliability(s_dir, policy) if reliability else make_inquire(s_dir, policy)
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
        # Baseline / flag-off: testimony ABI may be off — skip silently for baseline
        # that still uses plan_inquiry conflict path without storing claims.
        if hasattr(ag, "observe_testimony"):
            out = ag.observe_testimony(
                {
                    "speaker_token": row["speaker_token"],
                    "context_atoms": list(row["context_atoms"]),
                    "claim_atoms": list(row["claim_atoms"]),
                    "event_token": row["event_token"],
                }
            )
            # Off → reliability_off is expected
            return
        return
    out = ag.observe_testimony(
        {
            "speaker_token": row["speaker_token"],
            "context_atoms": list(row["context_atoms"]),
            "claim_atoms": list(row["claim_atoms"]),
            "event_token": row["event_token"],
        }
    )
    if not out.get("ok"):
        raise RuntimeError(f"testimony reject: {out}")


def clear_verification_rows(s_dir: Path) -> int:
    """Strip experience_reliability and verification-eligible grounding (evt_* + provenance)."""
    n = clear_by_source(s_dir, SOURCE_RELIABILITY)
    for p in sorted(s_dir.glob("*.tag")):
        _fid, tags = parse_tagfile(p.read_text(encoding="utf-8"))
        if str(tags.get("source") or "") != SOURCE_GROUND:
            continue
        prov = str(tags.get("provenance") or "")
        tid = str(tags.get("trial_id") or "")
        if prov in {"direct", "experiment", "state_read"} or tid.startswith("evt_"):
            p.unlink()
            n += 1
    return n


def clear_consequence_grounds(s_dir: Path) -> int:
    n = 0
    for p in sorted(s_dir.glob("*.tag")):
        _fid, tags = parse_tagfile(p.read_text(encoding="utf-8"))
        if str(tags.get("source") or "") != SOURCE_GROUND:
            continue
        tid = str(tags.get("trial_id") or "")
        if tid.startswith("cons_") or tid.startswith("conflict_"):
            p.unlink()
            n += 1
    return n


def verify_baseline_prereg() -> tuple[bool, str, dict[str, Any]]:
    if not PREREG_BASELINE.exists():
        return False, "missing baseline prereg", {}
    lock = json.loads(PREREG_BASELINE.read_text(encoding="utf-8"))
    if lock.get("lab") != "TM.0.20.RELIABILITY":
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
    if pins.get("inquire.lock") != _sha_file(INQUIRE_LOCK):
        return False, "prior inquire.lock pin", lock
    return True, "reliability_baseline.prereg.lock intact", lock


def verify_mech_prereg() -> tuple[bool, str, dict[str, Any]]:
    if not PREREG_MECH.exists():
        return False, "missing mech prereg", {}
    lock = json.loads(PREREG_MECH.read_text(encoding="utf-8"))
    if lock.get("lab") != "TM.0.20.RELIABILITY.MECH":
        return False, "lab drift", lock
    if lock.get("earned_next") is not False or lock.get("ex0s") is not None:
        return False, "earn/product drift", lock
    if lock.get("preregistered_claim") != MECH_CLAIM:
        return False, "claim drift", lock
    if lock.get("flag") != "use_source_reliability" or lock.get("flag_default") is not False:
        return False, "flag contract", lock
    if lock.get("lambda") != 4 or lock.get("n_min") != 2:
        return False, "arithmetic contract", lock
    if lock.get("fixture_sha") != _sha_file(FIXTURE_JSON):
        return False, "fixture_sha pin", lock
    if any(k in lock for k in ("agent_sha", "run_tm020reliability_sha", "make_reliability_sha")):
        return False, "prereg contains runner/agent SHAs", lock
    return True, "reliability_mech.prereg.lock intact", lock


def verify_wall_prereg() -> tuple[bool, str, dict[str, Any]]:
    if not PREREG_WALL.exists():
        return False, "missing wall prereg", {}
    lock = json.loads(PREREG_WALL.read_text(encoding="utf-8"))
    if lock.get("lab") != "TM.0.20.RELIABILITY.WALL":
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
    dims = set(lock.get("social_dimensions") or [])
    need = {"competence", "honesty", "access", "stability", "independence", "intent"}
    if not need.issubset(dims):
        return False, "social dimensions incomplete", lock
    return True, "reliability_wall.prereg.lock intact", lock


def _margin(ag: Any, speaker: str, ctx: Sequence[str]) -> float:
    return float(ag.source_evidence_margin(speaker, list(ctx)))


def _reliability_count(
    ag: Any, speaker: str | None = None, derived: str | None = None
) -> int:
    n = 0
    for rec in ag._reliability_rows():
        if speaker and str(rec.tags.get("speaker_token") or "") != speaker:
            continue
        if derived and str(rec.tags.get("derived") or "") != derived:
            continue
        n += 1
    return n


def run_ops(
    ag: Any,
    s_dir: Path,
    tmp: Path,
    policy: UsePolicy,
    ops: Sequence[dict[str, Any]],
    *,
    reliability: bool,
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
        if kind == "liveness_check":
            if first_fail is not None:
                continue
            got = bool(
                ag._inquire_already_resolved_factor(str(op["cue"]), str(op["factor"]))
            )
            expect = bool(op.get("expect_resolved"))
            if got != expect:
                first_fail = {
                    "stage": op.get("stage"),
                    "probe": op.get("id") or "liveness_check",
                    "expected": expect,
                    "actual": got,
                    "failure_family": op.get("failure_family") or "inquiry_strategy",
                }
            else:
                st = op.get("stage")
                if st in STAGES:
                    last_clear = st
            continue
        if kind == "ground" or (kind is None and "symbol" in op and "paired" in op):
            # allow bare ground rows
            row = op if "symbol" in op else op
            if row.get("op") == "ground" or "symbol" in row:
                apply_ground(ag, row if "symbol" in row else op)
            continue
        if kind == "testimony":
            apply_testimony(ag, op)
            continue
        if kind == "set_costs":
            ag.inquire_cost_ask = int(op["ask"])
            ag.inquire_cost_experiment = int(op["experiment"])
            continue
        if kind == "teach_renders":
            for pr in op.get("renders") or []:
                teach_probe_render(
                    ag,
                    context_atoms=list(pr.get("context_atoms") or op.get("context_atoms") or ["scene", "fac_lab"]),
                    probe_atoms=list(pr["probe_atoms"]),
                    prefix_id=str(pr.get("id") or "tr"),
                )
            continue
        if kind == "note_observed":
            ag.note_inquire_observation(
                context_atoms=list(op["context_atoms"]),
                input_symbols=list(op["input_symbols"]),
                probe_kind=str(op.get("probe_kind") or "ask"),
                probe_atoms=list(op["probe_atoms"]),
                cost=int(op.get("cost") or 2),
            )
            continue
        if kind == "strip_consequence_rows":
            clear_consequence_grounds(s_dir)
            reload_store(ag)
            continue
        if kind == "margin_check":
            if first_fail is not None:
                continue
            m = _margin(ag, op["speaker"], op["context_atoms"])
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
                    "failure_family": op.get("failure_family") or "calibration",
                }
            else:
                st = op.get("stage")
                if st in STAGES:
                    last_clear = st
            continue
        if kind == "reliability_count":
            if first_fail is not None:
                continue
            n = _reliability_count(ag, op.get("speaker"), op.get("derived"))
            ok = n >= int(op.get("expect_ge") or 1)
            if not ok:
                first_fail = {
                    "stage": op.get("stage"),
                    "probe": op.get("id") or "reliability_count",
                    "expected": op.get("expect_ge"),
                    "actual": n,
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
            if op.get("set_costs"):
                sc = op["set_costs"]
                ag.inquire_cost_ask = int(sc["ask"])
                ag.inquire_cost_experiment = int(sc["experiment"])
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
            if op.get("allow_hold") and plan.get("status") == "HOLD":
                ok = True
            if not ok:
                first_fail = {
                    "stage": op.get("stage"),
                    "probe": op.get("id"),
                    "expected": op.get("expect_status"),
                    "actual": plan.get("status"),
                    "answer": plan.get("answer_symbols"),
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
            fail = run_fork(tmp, s_dir, policy, op, reliability=reliability)
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
        # Bare ground rows embedded in lists (from generator)
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


def run_fork(
    tmp: Path,
    s_dir: Path,
    policy: UsePolicy,
    fork: dict[str, Any],
    *,
    reliability: bool,
) -> dict[str, Any] | None:
    kind = fork["kind"]
    clone = tmp / f"fork_{kind}_{fork.get('stage')}_{fork.get('id') or 'x'}"
    if clone.exists():
        shutil.rmtree(clone)
    shutil.copytree(s_dir, clone)
    ag = make_reliability(clone, policy) if reliability else make_inquire(clone, policy)
    reload_store(ag)
    ag.reset_rho()
    if hasattr(ag, "reset_inquire_budget"):
        ag.reset_inquire_budget()
    if kind == "strip_verification":
        clear_verification_rows(clone)
        reload_store(ag)
    elif kind == "strip_testimony":
        clear_by_source(clone, SOURCE_TESTIMONY)
        reload_store(ag)
    elif kind == "strip_reliability_keep_testimony":
        clear_by_source(clone, SOURCE_RELIABILITY)
        reload_store(ag)
    elif kind == "donor_calibration":
        # Empty organism: inject only donor calibration (reliability rows + matching
        # past testimony context), then current conflict testimony — no host verify
        # of the current target.
        clear_by_source(clone, SOURCE_TESTIMONY)
        clear_by_source(clone, SOURCE_RELIABILITY)
        clear_by_source(clone, SOURCE_GROUND)
        clear_by_source(clone, SOURCE_INQUIRE)
        reload_store(ag)
        ensure_context_grounded(
            ag, list(fork.get("context_atoms") or ["scene", "fac_lab"]), tag="donor"
        )
        for row in fork.get("donor_ops") or []:
            if row.get("op") == "testimony":
                apply_testimony(ag, row)
            elif row.get("op") == "ground" or "symbol" in row:
                apply_ground(ag, row if "op" in row else {**row, "op": "ground"})
        for row in fork.get("conflict_ops") or []:
            if row.get("op") == "testimony":
                apply_testimony(ag, row)
    else:
        raise ValueError(kind)
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


def flatten_life_ops(fix: dict[str, Any]) -> list[dict[str, Any]]:
    ops: list[dict[str, Any]] = []
    life = fix.get("life") or {}
    for stage in STAGES:
        block = life.get(stage) or {}
        for op in block.get("ops") or []:
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
    with tempfile.TemporaryDirectory(prefix="tm020rel_base_") as tmp:
        s_dir, ag = fresh(Path(tmp), "baseline", policy, reliability=False)
        assert not getattr(ag, "use_source_reliability", False)
        # Conflicting teachers without reliability weighting → HOLD
        # Baseline script may include testimony ops (no-ops when off) + plan
        result = run_ops(
            ag, s_dir, Path(tmp), policy, fixture["script_baseline"], reliability=False
        )
        # Also ensure plan_inquiry does not invent ANSWER from testimony when off
        apply_testimony(
            ag,
            {
                "speaker_token": "spk_a",
                "context_atoms": ["scene", "fac_lab"],
                "claim_atoms": ["curr_b0", "hyp_red"],
                "event_token": "eb0a",
            },
        )
        apply_testimony(
            ag,
            {
                "speaker_token": "spk_b",
                "context_atoms": ["scene", "fac_lab"],
                "claim_atoms": ["curr_b0", "hyp_blue"],
                "event_token": "eb0b",
            },
        )
        ok, plan = score_plan(
            ag,
            {
                "context_atoms": ["scene", "fac_lab"],
                "input_symbols": ["what", "curr_b0"],
                "expect_status": "HOLD",
            },
        )
        result_ok = result["ok"] and ok and plan.get("status") == "HOLD"
        first_fail = result["first_fail"]
        if not ok:
            first_fail = {
                "probe": "B_conflict_recheck",
                "expected": "HOLD",
                "actual": plan.get("status"),
                "why": plan.get("why"),
            }
    summary = {
        "version": "TM.0.20.RELIABILITY.BASELINE",
        "lab": "TM.0.20.RELIABILITY",
        "phase": "A",
        "ok": result_ok,
        "earned_next": False,
        "ex0s": None,
        "claim": BASELINE_CLAIM,
        "first_fail": first_fail,
        "fixture_sha": _sha_file(FIXTURE_JSON),
        "agent_sha": _sha_file(AGENT_PY),
        "factory": "make_inquire",
    }
    if write_lock:
        snap = {
            "version": "TM.0.20.RELIABILITY.BASELINE",
            "lab": "TM.0.20.RELIABILITY",
            "phase": "A",
            "ex0s_under_test": "0.0.004",
            "earned_next": False,
            "ex0s": None,
            "ok": summary["ok"],
            "first_fail": summary["first_fail"],
            "fixture_sha": _sha_file(FIXTURE_JSON),
            "reliability_baseline_prereg_sha": _sha_file(PREREG_BASELINE),
            "inquire_lock_sha": _sha_file(INQUIRE_LOCK),
            "agent_sha": _sha_file(AGENT_PY),
            "run_tm020reliability_sha": _sha_file(Path(__file__)),
            "refuse": [
                "editing agent.py in Phase A",
                "host confirm|contradict",
                "earned_next=true or non-null ex0s",
            ],
        }
        BASELINE_LOCK.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return summary


def write_candidate_lock() -> dict[str, Any]:
    ok_w, why_w, _ = verify_wall_prereg()
    if not ok_w:
        raise RuntimeError(f"wall prereg required before candidate: {why_w}")
    snap = {
        "version": "TM.0.20.RELIABILITY.CANDIDATE",
        "lab": "TM.0.20.RELIABILITY.MECH",
        "ex0s_under_test": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "flag": "use_source_reliability",
        "sources": [SOURCE_TESTIMONY, SOURCE_RELIABILITY],
        "margin_name": "source_evidence_margin",
        "observation_abi": "observe_testimony",
        "plan_abi": "plan_inquiry",
        "factory": "experiments.run_tm020reliability.make_reliability",
        "lambda": 4,
        "n_min": 2,
        "jaccard": 0.5,
        "r10_liveness": "opt-in via use_source_reliability only",
        "agent_sha": _sha_file(AGENT_PY),
        "observe_testimony_sha": _sha_src(agent_mod.ThreeMemoryAgent.observe_testimony),
        "source_evidence_margin_sha": _sha_src(
            agent_mod.ThreeMemoryAgent.source_evidence_margin
        ),
        "make_reliability_sha": _sha_src(make_reliability),
        "run_tm020reliability_sha": _sha_file(Path(__file__)),
        "reliability_mech_prereg_sha": _sha_file(PREREG_MECH),
        "reliability_baseline_sha": _sha_file(BASELINE_LOCK) if BASELINE_LOCK.exists() else None,
        "reliability_wall_prereg_sha": _sha_file(PREREG_WALL),
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
    with tempfile.TemporaryDirectory(prefix="tm020rel_smoke_") as tmp:
        _, ag = fresh(Path(tmp), "smoke", policy, reliability=True)
        bad = ag.observe_testimony({"speaker_token": "a"})
        if bad.get("why") != "exact_key_reject":
            return {"ok": False, "why": f"reject smoke failed: {bad}"}
        out = ag.observe_testimony(
            {
                "speaker_token": "spk_smoke",
                "context_atoms": ["scene", "fac_lab"],
                "claim_atoms": ["past_smoke", "hyp_ok"],
                "event_token": "esmoke",
            }
        )
        if not out.get("ok"):
            return {"ok": False, "why": f"testimony failed: {out}"}
        apply_ground(
            ag,
            {
                "symbol": "past_smoke",
                "paired": "hyp_ok",
                "trial_id": "evt_esmoke__v0",
                "result": "success",
                "provenance": "direct",
            },
        )
        if _reliability_count(ag, "spk_smoke") < 1:
            return {"ok": False, "why": "no derived reliability row"}
        _ = ag.plan_inquiry(
            {"context_atoms": ["scene", "fac_lab"], "input_symbols": ["what", "curr_x"]}
        )
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
    with tempfile.TemporaryDirectory(prefix="tm020rel_cells_") as tmp:
        root = Path(tmp)
        for name, cell in fixture["unit_cells"].items():
            s_dir = root / name
            empty_birth(s_dir)
            use_rel = bool(cell.get("flag"))
            ag = make_reliability(s_dir, policy) if use_rel else make_inquire(s_dir, policy)
            ag.reset_rho()
            if hasattr(ag, "reset_inquire_budget"):
                ag.reset_inquire_budget()
            ensure_context_grounded(
                ag, list(cell.get("probe", {}).get("context_atoms") or ["scene"]), tag=name
            )
            why = "ok"
            ok = True
            if cell.get("malformed") is not None:
                bad = ag.observe_testimony(cell["malformed"])
                ok = bad.get("why") == "exact_key_reject"
                why = bad.get("why")
                cells_out.append(
                    {"cell": name, "ok": ok, "actual": "exact_key_reject", "why": why if ok else f"got {bad}"}
                )
                continue
            for row in cell.get("ops") or []:
                if row.get("op") == "testimony":
                    apply_testimony(ag, row)
                elif row.get("op") == "ground" or "symbol" in row:
                    apply_ground(ag, row if "op" in row else {**row, "op": "ground"})
            poke = cell["probe"]
            cell_ok, plan = score_plan(ag, poke)
            actual = plan.get("status")
            if not use_rel:
                # flag off: must not answer via source_evidence_margin
                if plan.get("why") == "source_evidence_margin":
                    cell_ok = False
                    why = "flag_should_be_off"
            if cell_ok and cell.get("forks"):
                for fork in cell["forks"]:
                    f = dict(fork)
                    f.setdefault("context_atoms", poke["context_atoms"])
                    f.setdefault("input_symbols", poke["input_symbols"])
                    f.setdefault("stage", name)
                    fail = run_fork(root, s_dir, policy, f, reliability=use_rel)
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
        "version": "TM.0.20.RELIABILITY.MECH",
        "lab": "TM.0.20.RELIABILITY.MECH",
        "ex0s_under_test": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "ok": cells["ok"],
        "n_pass": cells["n_pass"],
        "n_cells": cells["n_cells"],
        "cells": cells["cells"],
        "agent_sha": _sha_file(AGENT_PY),
        "candidate_sha": _sha_file(CANDIDATE_LOCK),
        "run_tm020reliability_sha": _sha_file(Path(__file__)),
    }
    MECH_LOCK.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return snap


def run_capacity_lanes(*, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    fixture = load_fixture()
    policy = UsePolicy(seed=seed)
    lanes_out: dict[str, Any] = {}
    all_ok = True
    with tempfile.TemporaryDirectory(prefix="tm020rel_cap_") as tmp:
        root = Path(tmp)
        for lane_name, lane in (fixture.get("capacity") or {}).items():
            scored = set(lane.get("scored_rungs") or lane.get("rungs") or [])
            metric_only = set(lane.get("metric_only_rungs") or [])
            rungs_out = []
            first_fail_rung = None
            for branch in lane.get("branches") or []:
                rung = branch["rung"]
                s_dir, ag = fresh(root, f"{lane_name}_{rung}", policy, reliability=True)
                t0 = time.perf_counter()
                result = run_ops(
                    ag, s_dir, root, policy, branch["script"], reliability=True
                )
                dt = time.perf_counter() - t0
                metrics = dict(result.get("metrics") or {})
                metrics["wall_s"] = dt
                metrics["rows_examined"] = metrics.get("s_row_count")
                is_metric_only = bool(branch.get("metric_only")) or rung in metric_only
                # Metric-only still records true plan outcome; never force ok=True.
                ok = bool(result["ok"])
                if not ok and first_fail_rung is None and (
                    rung in scored or not is_metric_only
                ):
                    # Metric-only rungs do not fail the lane, but must not claim pass
                    if not is_metric_only:
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
            lane_ok = first_fail_rung is None
            # Lane ok requires all non-metric-only rungs ok
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
    with tempfile.TemporaryDirectory(prefix="tm020rel_life_") as tmp:
        root = Path(tmp)
        s_dir, ag = fresh(root, "life", policy, reliability=True)
        # Accumulating organism across R0–R12
        life_ops = flatten_life_ops(fixture)
        main = run_ops(ag, s_dir, root, policy, life_ops, reliability=True)
        # Twin
        s_twin, ag_twin = fresh(root, "twin", policy, reliability=True)
        twin = run_ops(
            ag_twin, s_twin, root, policy, fixture.get("script_twin") or [], reliability=True
        )
        capacity = run_capacity_lanes(seed=seed)
    life_last = main.get("last_stage_clear")
    first_fail = main.get("first_fail")
    # If life cleared through R12 capacity_launch marker even if last plan was earlier
    if main["ok"]:
        life_last = "R12"
    last = "R12" if main["ok"] and capacity.get("ok") else life_last
    summary = {
        "version": "TM.0.20.RELIABILITY",
        "lab": "TM.0.20.RELIABILITY",
        "ex0s_under_test": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "ok": first_fail is None and twin.get("ok") and capacity.get("ok"),
        "last_stage_clear": last,
        "life_last_stage_clear": life_last if main["ok"] else life_last,
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
        "run_tm020reliability_sha": _sha_file(Path(__file__)),
        "bounded_claim": (
            "Context-conditioned predictive reliability learned from independently "
            "verified outcomes."
        ),
    }
    if write_lock:
        write_reliability_lock(summary)
        write_results_md(summary)
    return summary


def write_reliability_lock(summary: dict[str, Any]) -> dict[str, Any]:
    snap = {
        "version": "TM.0.20.RELIABILITY",
        "lab": "TM.0.20.RELIABILITY",
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
        "run_tm020reliability_sha": summary.get("run_tm020reliability_sha"),
        "bounded_claim": summary.get("bounded_claim"),
        "refuse": [
            "trust_score",
            "host confirm|contradict",
            "collapsing six social dimensions",
            "earned_next=true or non-null ex0s",
        ],
    }
    RELIABILITY_LOCK.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return snap


def write_results_md(summary: dict[str, Any]) -> None:
    cap = summary.get("capacity") or {}
    lines = [
        "# TM.0.20.RELIABILITY results: evidence-source calibration",
        "",
        "**Ex0S under test:** **0.0.004** (not a new stamp)",
        "**Lab:** TM.0.20.RELIABILITY",
        f"**ok:** `{summary.get('ok')}`",
        f"**life_last_stage_clear:** `{summary.get('life_last_stage_clear')}`",
        f"**first_fail:** `{summary.get('first_fail')}`",
        "",
        "## Bounded claim",
        "",
        "> Context-conditioned predictive reliability learned from independently verified outcomes.",
        "",
        "Expanded: Ex0S derived a context-conditioned **source_evidence_margin** (predictive accuracy) "
        "from organism-compared claim↔independent-outcome evidence, used it to weight later "
        "non-duplicated testimony conflicts, retained uncertainty without sufficient verification, "
        "and revised the margin as append-only evidence accumulated—without claiming honesty, "
        "trustworthiness, access, independence, or intent.",
        "",
        "## Capacity",
        "",
        f"- ok: `{cap.get('ok')}`",
        "",
        "## Explicit non-claims",
        "",
        "- Not a `trust_score`",
        "- Does not implement honesty / access / independence / intent models",
        "- Wall probes those dimensions diagnostically only",
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
    with tempfile.TemporaryDirectory(prefix="tm020rel_wall_") as tmp:
        root = Path(tmp)
        scripts = fixture.get("wall_scripts") or {}
        for w in fixture["wall"]:
            wid = w["id"]
            note = w.get("note") or ""
            if wid in scripts:
                s_dir, ag = fresh(root, wid, policy, reliability=True)
                out = run_ops(
                    ag, s_dir, root, policy, scripts[wid]["ops"], reliability=True
                )
                if scripts[wid].get("expect_fail"):
                    # Script encodes predictive-accuracy behavior. Succeeding it
                    # exposes the honesty/social gap → diagnostic wall fail.
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
        "version": "TM.0.20.RELIABILITY.WALL",
        "lab": "TM.0.20.RELIABILITY.WALL",
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
        "reliability_wall_prereg_sha": _sha_file(PREREG_WALL),
        "reliability_lock_sha": _sha_file(RELIABILITY_LOCK) if RELIABILITY_LOCK.exists() else None,
        "agent_sha": _sha_file(AGENT_PY),
        "run_tm020reliability_sha": _sha_file(Path(__file__)),
        "note": (
            "Results only. Predictive accuracy earned in-life; multi-factor source model "
            "remains open. not_run probes are not first_fail diagnostics. "
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
        print(json.dumps(run_life(seed=args.seed, write_lock=args.write_lock), indent=2, default=str))
        return

    if args.wall or args.write_wall:
        print(json.dumps(run_wall(seed=args.seed, write_lock=args.write_wall), indent=2, default=str))
        return

    ap.print_help()


if __name__ == "__main__":
    main()
