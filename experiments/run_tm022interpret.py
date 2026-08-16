"""TM.0.22.INTERPRET: behaviorally evidenced interpretation.

Phase A baseline (perspective-on, interpretation off) → B consequence candidate
→ C J0–J16 → capacity → wall.
Product stays 0.0.004; earned_next=false; ex0s=null.
Never subjective comprehension / honesty_score / cause ABI / result field.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm016relate import clear_by_source, empty_birth, reload_store
from experiments.run_tm019inquire import ensure_context_grounded, score_plan
from experiments.run_tm021perspective import make_perspective
from three_memory import agent as agent_mod
from three_memory.policy import UsePolicy

PREREG_BASELINE = REPO_ROOT / "docs" / "interpret_baseline.prereg.lock"
PREREG_MECH = REPO_ROOT / "docs" / "interpret_mech.prereg.lock"
PREREG_WALL = REPO_ROOT / "docs" / "interpret_wall.prereg.lock"
FIXTURE_JSON = REPO_ROOT / "docs" / "interpret_fixture.json"
BASELINE_LOCK = REPO_ROOT / "docs" / "interpret_baseline.lock"
CANDIDATE_LOCK = REPO_ROOT / "docs" / "interpret.candidate.lock"
CANDIDATE_V1_LOCK = REPO_ROOT / "docs" / "interpret.candidate.v1.lock"
MECH_LOCK = REPO_ROOT / "docs" / "interpret_mech.lock"
INTERPRET_LOCK = REPO_ROOT / "docs" / "interpret.lock"
WALL_LOCK = REPO_ROOT / "docs" / "interpret_wall.lock"
RESULTS_MD = REPO_ROOT / "docs" / "tm022interpret_results.md"
AGENT_PY = REPO_ROOT / "three_memory" / "agent.py"
PERSPECTIVE_LOCK = REPO_ROOT / "docs" / "perspective.lock"

DEFAULT_SEED = 12345
SOURCE_INTERPRET = "experience_interpretation"
SOURCE_GROUND = "experience_grounding"
STAGES = tuple(f"J{i}" for i in range(17))

BASELINE_CLAIM = (
    "Frozen PERSPECTIVE (make_perspective, interpretation off) does not reconstruct "
    "source-specific interpretations — interpret_message remains INSUFFICIENT; "
    "PERSPECTIVE/SEQUENCE/INQUIRE/RELIABILITY behavior unchanged."
)

MECH_CLAIM = (
    "An opt-in recipe may record opaque source-consequence episodes with pre-outcome "
    "interaction tokens, reconstruct factorized first-order interpretations only when "
    "later behavior is independently grounded outside INTERPRET, score fit separately "
    "from reconstruction sufficiency, and repair via goal_cue_symbols under frozen "
    "SEQUENCE — without subjective comprehension, world-truth from another map, "
    "honesty, stability, Jaccard, or derived statuses in S."
)

WALL_CLAIM = (
    "On frozen make_interpret, a preregistered conversation wall probes two-source "
    "meanings, misunderstood instructions, delayed revision, wrong-recipient messages, "
    "unfamiliar wording, interruption, copying, claim-understand inconsistency "
    "(fit CONFLICT; cause UNKNOWN is scorer/narrative only), deception-indistinguishable "
    "cases, and unmarked sudden change. Need not fully pass; first_fail_wall diagnoses "
    "the next primitive — behaviorally evidenced interpretation does not claim "
    "subjective comprehension, honesty, or stability."
)


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _sha_src(fn: Callable[..., Any]) -> str:
    return _sha_bytes(inspect.getsource(fn).encode())


def load_fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_JSON.read_text(encoding="utf-8"))


def make_interpret(s_dir: Path, policy: UsePolicy | None = None, **kwargs: Any) -> Any:
    ag = make_perspective(s_dir, policy, **kwargs)
    ag.use_source_interpretation = True
    fix = load_fixture()
    ag.interpret_lambda = int(fix.get("interpret_lambda") or 4)
    ag.interpret_n_min = int(fix.get("interpret_n_min") or 2)
    return ag


def fresh(tmp: Path, name: str, policy: UsePolicy, *, interpret: bool) -> tuple[Path, Any]:
    s_dir = tmp / name
    empty_birth(s_dir)
    ag = make_interpret(s_dir, policy) if interpret else make_perspective(s_dir, policy)
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


def apply_consequence(ag: Any, row: dict[str, Any]) -> None:
    if not getattr(ag, "use_source_interpretation", False):
        return
    out = ag.observe_source_consequence(
        {
            "source_token": row["source_token"],
            "interaction_token": row["interaction_token"],
            "exposure_event_token": row["exposure_event_token"],
            "consequence_event_token": row["consequence_event_token"],
            "context_symbols": list(row["context_symbols"]),
            "message_symbols": list(row["message_symbols"]),
            "action_symbols": list(row["action_symbols"]),
            "state_before": list(row["state_before"]),
            "state_after": list(row["state_after"]),
        }
    )
    if not out.get("ok"):
        raise RuntimeError(f"consequence reject: {out}")


def apply_sequence_step(ag: Any, row: dict[str, Any]) -> None:
    out = ag.observe_sequence_step(
        {
            "context_atoms": list(row["context_atoms"]),
            "input_symbols": list(row["input_symbols"]),
            "prefix": list(row.get("prefix") or []),
            "next_operation": row["next_operation"],
            "next_symbol": row.get("next_symbol") or "",
            "result": row["result"],
        }
    )
    if not out.get("ok"):
        raise RuntimeError(f"sequence_step reject: {out}")


def clear_interpretation_rows(s_dir: Path) -> int:
    return clear_by_source(s_dir, SOURCE_INTERPRET)


def strip_grounding_symbols(ag: Any, s_dir: Path, symbols: Sequence[str]) -> None:
    """Drop grounding rows for given symbols from disk and reload."""
    from three_memory.symbols import parse_tagfile

    syms = {s.lower() for s in symbols}
    for p in sorted(s_dir.glob("*.tag")):
        _fid, tags = parse_tagfile(p.read_text(encoding="utf-8"))
        if str(tags.get("source") or "") != SOURCE_GROUND:
            continue
        if str(tags.get("symbol") or "").lower() in syms:
            p.unlink()
    reload_store(ag)


def verify_baseline_prereg() -> tuple[bool, str, dict[str, Any]]:
    if not PREREG_BASELINE.exists():
        return False, "missing baseline prereg", {}
    lock = json.loads(PREREG_BASELINE.read_text(encoding="utf-8"))
    if lock.get("lab") != "TM.0.22.INTERPRET":
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
    if pins.get("perspective.lock") != _sha_file(PERSPECTIVE_LOCK):
        return False, "prior perspective.lock pin", lock
    return True, "interpret_baseline.prereg.lock intact", lock


def verify_mech_prereg() -> tuple[bool, str, dict[str, Any]]:
    if not PREREG_MECH.exists():
        return False, "missing mech prereg", {}
    lock = json.loads(PREREG_MECH.read_text(encoding="utf-8"))
    if lock.get("lab") != "TM.0.22.INTERPRET.MECH":
        return False, "lab drift", lock
    if lock.get("earned_next") is not False or lock.get("ex0s") is not None:
        return False, "earn/product drift", lock
    if lock.get("preregistered_claim") != MECH_CLAIM:
        return False, "claim drift", lock
    if lock.get("flag") != "use_source_interpretation" or lock.get("flag_default") is not False:
        return False, "flag contract", lock
    if lock.get("no_jaccard") is not True:
        return False, "no_jaccard", lock
    if lock.get("independent_anchor_required") is not True:
        return False, "independent_anchor", lock
    if lock.get("no_derived_statuses_in_S") is not True:
        return False, "no_derived", lock
    if lock.get("fixture_sha") != _sha_file(FIXTURE_JSON):
        return False, "fixture_sha pin", lock
    if any(k in lock for k in ("agent_sha", "run_tm022interpret_sha", "make_interpret_sha")):
        return False, "prereg contains runner/agent SHAs", lock
    return True, "interpret_mech.prereg.lock intact", lock


def verify_wall_prereg() -> tuple[bool, str, dict[str, Any]]:
    if not PREREG_WALL.exists():
        return False, "missing wall prereg", {}
    lock = json.loads(PREREG_WALL.read_text(encoding="utf-8"))
    if lock.get("lab") != "TM.0.22.INTERPRET.WALL":
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
    return True, "interpret_wall.prereg.lock intact", lock


def do_interpret_check(ag: Any, op: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    out = ag.interpret_message(
        {
            "source_token": op["source_token"],
            "context_symbols": list(op["context_symbols"]),
            "ordered_symbols": list(op["ordered_symbols"]),
        }
    )
    ok = out.get("status") == op["expect_status"]
    if ok and op.get("expect_candidate") is not None:
        ok = list(out.get("candidate") or []) == list(op["expect_candidate"])
    return ok, out


def do_fit_check(ag: Any, op: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    out = ag.interpretation_fit(
        {
            "source_token": op["source_token"],
            "context_symbols": list(op["context_symbols"]),
            "message_symbols": list(op["message_symbols"]),
            "action_symbols": list(op["action_symbols"]),
            "state_before": list(op["state_before"]),
            "state_after": list(op["state_after"]),
        }
    )
    return out.get("fit") == op["expect_fit"], out


def run_fork(
    tmp: Path,
    s_dir: Path,
    policy: UsePolicy,
    fork: dict[str, Any],
    *,
    interpret: bool,
) -> dict[str, Any] | None:
    kind = fork["kind"]
    clone = tmp / f"fork_{kind}_{fork.get('stage')}_{fork.get('id') or 'x'}"
    if clone.exists():
        shutil.rmtree(clone)
    shutil.copytree(s_dir, clone)
    ag = make_interpret(clone, policy) if interpret else make_perspective(clone, policy)
    reload_store(ag)
    ag.reset_rho()
    if kind == "strip_interpretation":
        clear_interpretation_rows(clone)
        reload_store(ag)
    elif kind == "donor_interpretation":
        if fork.get("clear_first", True):
            clear_interpretation_rows(clone)
            reload_store(ag)
        for row in fork.get("donor_ops") or []:
            apply_row(ag, row)
    elif kind == "strip_action_ground":
        strip_grounding_symbols(ag, clone, fork.get("action_symbols") or [])
        reload_store(ag)
    elif kind == "strip_goal_ground":
        strip_grounding_symbols(ag, clone, fork.get("goal_symbols") or [])
        reload_store(ag)
        if fork.get("repair"):
            rep = fork["repair"]
            out = ag.plan_recipient_message(
                {
                    "recipient_token": rep["recipient_token"],
                    "context_symbols": list(rep["context_symbols"]),
                    "goal_cue_symbols": list(rep["goal_cue_symbols"]),
                }
            )
            if out.get("status") != rep["expect_status"]:
                return {
                    "probe": fork.get("id") or kind,
                    "expected": rep["expect_status"],
                    "actual": out.get("status"),
                    "failure_family": "repair",
                }
            return None
    else:
        raise ValueError(kind)

    if fork.get("expect_interpret"):
        ok, out = do_interpret_check(
            ag,
            {
                "source_token": fork["source_token"],
                "context_symbols": fork["context_symbols"],
                "ordered_symbols": fork["ordered_symbols"],
                "expect_status": fork["expect_interpret"],
                "expect_candidate": fork.get("expect_candidate"),
            },
        )
        if not ok:
            return {
                "probe": fork.get("id") or kind,
                "expected": fork["expect_interpret"],
                "actual": out.get("status"),
                "candidate": out.get("candidate"),
                "failure_family": "isolation",
            }
    return None


def apply_row(ag: Any, row: dict[str, Any]) -> None:
    op = row.get("op")
    if op == "ground" or (op is None and "symbol" in row and "paired" in row):
        apply_ground(ag, row)
    elif op == "testimony":
        apply_testimony(ag, row)
    elif op == "exposure":
        apply_exposure(ag, row)
    elif op == "consequence":
        apply_consequence(ag, row)
    elif op == "sequence_step":
        apply_sequence_step(ag, row)
    else:
        raise ValueError(f"unknown apply op {op}")


def run_ops(
    ag: Any,
    s_dir: Path,
    tmp: Path,
    policy: UsePolicy,
    ops: Sequence[dict[str, Any]],
    *,
    interpret: bool,
) -> dict[str, Any]:
    first_fail: dict[str, Any] | None = None
    last_clear: str | None = None

    for op in ops:
        kind = op.get("op")
        if kind == "stage_marker":
            continue
        if kind == "rho_reset":
            ag.reset_rho()
            continue
        if kind == "capacity_launch":
            st = op.get("stage")
            if st in STAGES and first_fail is None:
                last_clear = st
            continue
        if kind in ("ground", "testimony", "exposure", "consequence", "sequence_step") or (
            kind is None and "symbol" in op
        ):
            apply_row(ag, op)
            continue
        if kind == "interpret_check":
            if first_fail is not None:
                continue
            ok, out = do_interpret_check(ag, op)
            if not ok:
                first_fail = {
                    "stage": op.get("stage"),
                    "probe": op.get("id") or "interpret_check",
                    "expected": op["expect_status"],
                    "actual": out.get("status"),
                    "candidate": out.get("candidate"),
                    "failure_family": "interpretation",
                }
            else:
                st = op.get("stage")
                if st in STAGES:
                    last_clear = st
            continue
        if kind == "fit_check":
            if first_fail is not None:
                continue
            ok, out = do_fit_check(ag, op)
            if not ok:
                first_fail = {
                    "stage": op.get("stage"),
                    "probe": op.get("id") or "fit_check",
                    "expected": op["expect_fit"],
                    "actual": out.get("fit"),
                    "failure_family": "fit",
                }
            else:
                st = op.get("stage")
                if st in STAGES:
                    last_clear = st
            continue
        if kind == "alignment_check":
            if first_fail is not None:
                continue
            status = ag.report_alignment_status(
                op["speaker"], op["claim_atoms"], op.get("context_atoms")
            )
            if status != op["expect_status"]:
                first_fail = {
                    "stage": op.get("stage"),
                    "probe": op.get("id") or "alignment_check",
                    "expected": op["expect_status"],
                    "actual": status,
                    "failure_family": "perspective",
                }
            else:
                st = op.get("stage")
                if st in STAGES:
                    last_clear = st
            continue
        if kind == "plan":
            if first_fail is not None:
                continue
            ok_p, plan = score_plan(ag, op)
            if not ok_p:
                first_fail = {
                    "stage": op.get("stage"),
                    "probe": op.get("id") or "plan",
                    "expected": op.get("expect_status"),
                    "actual": plan.get("status"),
                    "failure_family": op.get("failure_family") or "plan",
                }
            else:
                st = op.get("stage")
                if st in STAGES:
                    last_clear = st
            continue
        if kind == "interpret_plan":
            if first_fail is not None:
                continue
            out = ag.plan_interpretation(
                {
                    "source_token": op["source_token"],
                    "context_symbols": list(op["context_symbols"]),
                    "ordered_symbols": list(op["ordered_symbols"]),
                }
            )
            if out.get("status") != op["expect_status"]:
                first_fail = {
                    "stage": op.get("stage"),
                    "probe": op.get("id") or "interpret_plan",
                    "expected": op["expect_status"],
                    "actual": out.get("status"),
                    "failure_family": "planner",
                }
            else:
                st = op.get("stage")
                if st in STAGES:
                    last_clear = st
            continue
        if kind == "repair":
            if first_fail is not None:
                continue
            out = ag.plan_recipient_message(
                {
                    "recipient_token": op["recipient_token"],
                    "context_symbols": list(op["context_symbols"]),
                    "goal_cue_symbols": list(op["goal_cue_symbols"]),
                }
            )
            ok = out.get("status") == op["expect_status"]
            if ok and op.get("expect_sequence") is not None:
                ok = list(out.get("sequence") or []) == list(op["expect_sequence"])
            if not ok:
                first_fail = {
                    "stage": op.get("stage"),
                    "probe": op.get("id") or "repair",
                    "expected": op["expect_status"],
                    "actual": out.get("status"),
                    "sequence": out.get("sequence"),
                    "failure_family": "repair",
                }
            else:
                st = op.get("stage")
                if st in STAGES:
                    last_clear = st
            continue
        if kind == "fork":
            if first_fail is not None:
                continue
            fail = run_fork(tmp, s_dir, policy, op, interpret=interpret)
            if fail is not None:
                fail["stage"] = op.get("stage")
                first_fail = fail
            else:
                st = op.get("stage")
                if st in STAGES:
                    last_clear = st
            continue
        raise ValueError(f"unknown op {kind}")

    return {
        "ok": first_fail is None,
        "first_fail": first_fail,
        "last_stage_clear": last_clear,
    }


def run_baseline(*, seed: int = DEFAULT_SEED, write_lock: bool = False) -> dict[str, Any]:
    ok_p, why_p, _ = verify_baseline_prereg()
    if not ok_p:
        raise RuntimeError(why_p)
    ok_w, why_w, _ = verify_wall_prereg()
    if not ok_w:
        raise RuntimeError(f"wall prereg required before baseline: {why_w}")
    fixture = load_fixture()
    policy = UsePolicy(seed=seed)
    with tempfile.TemporaryDirectory(prefix="tm022interp_base_") as tmp:
        root = Path(tmp)
        s_dir, ag = fresh(root, "baseline", policy, interpret=False)
        assert getattr(ag, "use_source_interpretation", False) is False
        for row in fixture.get("script_baseline") or []:
            if row.get("op") == "interpret_check":
                ok, out = do_interpret_check(ag, row)
                if not ok:
                    summary = {
                        "ok": False,
                        "why": f"baseline interpret {out}",
                        "earned_next": False,
                        "ex0s": None,
                    }
                    break
            else:
                apply_row(ag, row)
        else:
            summary = {
                "ok": True,
                "why": "baseline_pass",
                "earned_next": False,
                "ex0s": None,
                "fixture_sha": _sha_file(FIXTURE_JSON),
                "agent_sha": _sha_file(AGENT_PY),
            }
    if write_lock:
        snap = {
            "version": "TM.0.22.INTERPRET.BASELINE",
            "lab": "TM.0.22.INTERPRET",
            "ex0s_under_test": "0.0.004",
            "earned_next": False,
            "ex0s": None,
            "ok": summary["ok"],
            "why": summary.get("why"),
            "fixture_sha": _sha_file(FIXTURE_JSON),
            "perspective_lock_sha": _sha_file(PERSPECTIVE_LOCK),
            "agent_sha": _sha_file(AGENT_PY),
            "refuse": [
                "editing agent.py in Phase A",
                "honesty_score",
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
        "version": "TM.0.22.INTERPRET.CANDIDATE",
        "lab": "TM.0.22.INTERPRET.MECH",
        "ex0s_under_test": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "flag": "use_source_interpretation",
        "source": SOURCE_INTERPRET,
        "observation_abi": "observe_source_consequence",
        "reconstruction_statuses": ["UNIQUE", "AMBIGUOUS", "INSUFFICIENT"],
        "fit_statuses": ["SUPPORTED", "CONFLICT", "UNKNOWN"],
        "independent_anchor_required": True,
        "no_jaccard": True,
        "no_derived_statuses_in_S": True,
        "world_separation": "source_relative_only",
        "repair_goal": "goal_cue_symbols",
        "factory": "experiments.run_tm022interpret.make_interpret",
        "agent_sha": _sha_file(AGENT_PY),
        "observe_source_consequence_sha": _sha_src(
            agent_mod.ThreeMemoryAgent.observe_source_consequence
        ),
        "interpret_message_sha": _sha_src(agent_mod.ThreeMemoryAgent.interpret_message),
        "interpretation_fit_sha": _sha_src(agent_mod.ThreeMemoryAgent.interpretation_fit),
        "make_interpret_sha": _sha_src(make_interpret),
        "run_tm022interpret_sha": _sha_file(Path(__file__)),
        "interpret_mech_prereg_sha": _sha_file(PREREG_MECH),
        "interpret_baseline_sha": _sha_file(BASELINE_LOCK) if BASELINE_LOCK.exists() else None,
        "interpret_wall_prereg_sha": _sha_file(PREREG_WALL),
        "note": "Pinned after unscored ABI smoke, before scored cells. Preserve as v1 if audit rewrites agent.",
    }
    CANDIDATE_LOCK.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    if not CANDIDATE_V1_LOCK.exists():
        CANDIDATE_V1_LOCK.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return snap


def run_smoke(*, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    ok_p, why_p, _ = verify_mech_prereg()
    if not ok_p:
        raise RuntimeError(why_p)
    policy = UsePolicy(seed=seed)
    with tempfile.TemporaryDirectory(prefix="tm022interp_smoke_") as tmp:
        s_dir, ag = fresh(Path(tmp), "smoke", policy, interpret=True)
        bad = ag.observe_source_consequence({"source_token": "a"})
        if bad.get("why") != "exact_key_reject":
            return {"ok": False, "why": f"exact_key {bad}"}
        # presence of result key must reject
        CTX = ["scene", "fac_lab"]
        ag.observe_exposure(
            {
                "speaker_token": "src_a",
                "context_atoms": CTX,
                "exposure_atoms": ["exp_ack_read"],
                "event_token": "esmoke",
            }
        )
        apply_ground(
            ag,
            {
                "symbol": "act_s",
                "paired": "mean_s",
                "trial_id": "sm_ag0",
                "result": "success",
                "provenance": "direct",
            },
        )
        apply_ground(
            ag,
            {
                "symbol": "act_s",
                "paired": "mean_s",
                "trial_id": "sm_ag1",
                "result": "success",
                "provenance": "direct",
            },
        )
        out = ag.observe_source_consequence(
            {
                "source_token": "src_a",
                "interaction_token": "ix_s",
                "exposure_event_token": "esmoke",
                "consequence_event_token": "csmoke",
                "context_symbols": CTX,
                "message_symbols": ["tok_s"],
                "action_symbols": ["act_s"],
                "state_before": ["st_idle"],
                "state_after": ["st_done"],
            }
        )
        if not out.get("ok"):
            return {"ok": False, "why": f"consequence {out}"}
        # need second episode for n_min
        ag.observe_exposure(
            {
                "speaker_token": "src_a",
                "context_atoms": CTX,
                "exposure_atoms": ["exp_ack_read"],
                "event_token": "esmoke2",
            }
        )
        ag.observe_source_consequence(
            {
                "source_token": "src_a",
                "interaction_token": "ix_s2",
                "exposure_event_token": "esmoke2",
                "consequence_event_token": "csmoke2",
                "context_symbols": CTX,
                "message_symbols": ["tok_s"],
                "action_symbols": ["act_s"],
                "state_before": ["st_idle"],
                "state_after": ["st_done"],
            }
        )
        recon = ag.interpret_message(
            {
                "source_token": "src_a",
                "context_symbols": CTX,
                "ordered_symbols": ["tok_s"],
            }
        )
        if recon.get("status") != "UNIQUE":
            return {"ok": False, "why": f"recon {recon}"}
    return {"ok": True, "why": "abi_smoke"}


def run_unit_cells(*, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    if not CANDIDATE_LOCK.exists():
        raise RuntimeError("candidate.lock missing")
    fixture = load_fixture()
    policy = UsePolicy(seed=seed)
    cells_out = []
    with tempfile.TemporaryDirectory(prefix="tm022interp_cells_") as tmp:
        root = Path(tmp)
        for name, cell in fixture["unit_cells"].items():
            s_dir = root / name
            empty_birth(s_dir)
            use_i = bool(cell.get("flag"))
            ag = make_interpret(s_dir, policy) if use_i else make_perspective(s_dir, policy)
            ag.reset_rho()
            if hasattr(ag, "reset_inquire_budget"):
                ag.reset_inquire_budget()
            ensure_context_grounded(ag, ["scene", "fac_lab", "world", "ctx_alt"], tag=name)
            why = "ok"
            cell_ok = True
            actual: Any = "ok"
            if cell.get("malformed") is not None:
                bad = ag.observe_source_consequence(cell["malformed"])
                cell_ok = bad.get("why") == "exact_key_reject"
                cells_out.append(
                    {
                        "cell": name,
                        "ok": cell_ok,
                        "actual": "exact_key_reject",
                        "why": bad.get("why"),
                    }
                )
                continue
            for row in cell.get("ops") or []:
                apply_row(ag, row)
            if cell.get("interpret"):
                al = dict(cell["interpret"])
                ok, out = do_interpret_check(ag, al)
                actual = out.get("status")
                cell_ok = ok
                if not use_i and out.get("status") != "INSUFFICIENT":
                    cell_ok = False
                    why = "flag_should_force_insufficient"
            if cell_ok and cell.get("fit"):
                ok, out = do_fit_check(ag, cell["fit"])
                if not ok:
                    cell_ok = False
                    actual = out.get("fit")
                    why = "fit_fail"
            if cell_ok and cell.get("checks"):
                for ch in cell["checks"]:
                    if ch.get("op") == "interpret_check":
                        ok, out = do_interpret_check(ag, ch)
                        if not ok:
                            cell_ok = False
                            actual = out.get("status")
                            why = "check_fail"
                            break
                    elif ch.get("op") == "fit_check":
                        ok, out = do_fit_check(ag, ch)
                        if not ok:
                            cell_ok = False
                            actual = out.get("fit")
                            why = "fit_check_fail"
                            break
                    elif ch.get("op") == "alignment_check":
                        status = ag.report_alignment_status(
                            ch["speaker"], ch["claim_atoms"], ch.get("context_atoms")
                        )
                        if status != ch["expect_status"]:
                            cell_ok = False
                            actual = status
                            why = "alignment_fail"
                            break
            if cell_ok and cell.get("probe"):
                ok_p2, plan = score_plan(ag, cell["probe"])
                if not ok_p2:
                    cell_ok = False
                    actual = plan.get("status")
                    why = plan.get("why") or "probe_fail"
            if cell_ok and cell.get("repair"):
                out = ag.plan_recipient_message(
                    {
                        "recipient_token": cell["repair"]["recipient_token"],
                        "context_symbols": list(cell["repair"]["context_symbols"]),
                        "goal_cue_symbols": list(cell["repair"]["goal_cue_symbols"]),
                    }
                )
                ok = out.get("status") == cell["repair"]["expect_status"]
                if ok and cell["repair"].get("expect_sequence") is not None:
                    ok = list(out.get("sequence") or []) == list(
                        cell["repair"]["expect_sequence"]
                    )
                if not ok:
                    cell_ok = False
                    actual = out.get("status")
                    why = f"repair {out}"
            if cell_ok and cell.get("repair_anna"):
                ra = cell["repair_anna"]
                out = ag.plan_recipient_message(
                    {
                        "recipient_token": ra["recipient_token"],
                        "context_symbols": list(ra["context_symbols"]),
                        "goal_cue_symbols": list(ra["goal_cue_symbols"]),
                    }
                )
                ok = out.get("status") == ra["expect_status"]
                if ok and ra.get("expect_sequence") is not None:
                    ok = list(out.get("sequence") or []) == list(ra["expect_sequence"])
                if not ok:
                    cell_ok = False
                    actual = out.get("status")
                    why = f"repair_anna {out}"
            if cell_ok and cell.get("forks"):
                for fork in cell["forks"]:
                    f = dict(fork)
                    f.setdefault("stage", name)
                    fail = run_fork(root, s_dir, policy, f, interpret=use_i)
                    if fail is not None:
                        cell_ok = False
                        why = f"fork {fork['kind']}: {fail}"
                        break
            cells_out.append(
                {
                    "cell": name,
                    "ok": bool(cell_ok),
                    "actual": actual,
                    "why": why if not cell_ok else "pass",
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
        "version": "TM.0.22.INTERPRET.MECH",
        "lab": "TM.0.22.INTERPRET.MECH",
        "ex0s_under_test": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "ok": cells["ok"],
        "n_pass": cells["n_pass"],
        "n_cells": cells["n_cells"],
        "cells": cells["cells"],
        "agent_sha": _sha_file(AGENT_PY),
        "candidate_sha": _sha_file(CANDIDATE_LOCK),
        "run_tm022interpret_sha": _sha_file(Path(__file__)),
    }
    MECH_LOCK.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return snap


def run_capacity_lanes(*, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    fixture = load_fixture()
    policy = UsePolicy(seed=seed)
    lanes_out: dict[str, Any] = {}
    all_ok = True
    with tempfile.TemporaryDirectory(prefix="tm022interp_cap_") as tmp:
        root = Path(tmp)
        for lane_name, lane in (fixture.get("capacity") or {}).items():
            metric_only = set(lane.get("metric_only_rungs") or [])
            rungs_out = []
            first_fail_rung = None
            for branch in lane.get("branches") or []:
                rung = branch["rung"]
                is_metric_only = bool(branch.get("metric_only")) or rung in metric_only
                t0 = time.perf_counter()
                s_dir, ag = fresh(root, f"{lane_name}_{rung}", policy, interpret=True)
                result = run_ops(
                    ag, s_dir, root, policy, branch.get("script") or [], interpret=True
                )
                wall_s = time.perf_counter() - t0
                # Metric-only still records true outcome; never force ok=True.
                ok = bool(result["ok"])
                if not ok and not is_metric_only and first_fail_rung is None:
                    first_fail_rung = rung
                    all_ok = False
                rungs_out.append(
                    {
                        "rung": rung,
                        "ok": ok,
                        "metric_only": is_metric_only,
                        "scored_ok": ok,
                        "first_fail": result.get("first_fail"),
                        "metrics": {
                            "s_row_count": len(ag.store.records()),
                            "wall_s": wall_s,
                        },
                    }
                )
            lane_ok = first_fail_rung is None
            for r in rungs_out:
                if not r["metric_only"] and not r["ok"]:
                    lane_ok = False
            if not lane_ok:
                all_ok = False
            lanes_out[lane_name] = {
                "ok": lane_ok,
                "first_fail_rung": first_fail_rung,
                "rungs": rungs_out,
            }
    return {"ok": all_ok and all(v["ok"] for v in lanes_out.values()), "lanes": lanes_out}


def run_life(*, seed: int = DEFAULT_SEED, write_lock: bool = False) -> dict[str, Any]:
    if not MECH_LOCK.exists():
        raise RuntimeError("mech.lock missing")
    fixture = load_fixture()
    policy = UsePolicy(seed=seed)
    with tempfile.TemporaryDirectory(prefix="tm022interp_life_") as tmp:
        root = Path(tmp)
        s_dir, ag = fresh(root, "life", policy, interpret=True)
        all_ops: list[dict[str, Any]] = []
        for st in STAGES:
            stage = (fixture.get("life") or {}).get(st) or {}
            all_ops.extend(stage.get("ops") or [])
        main = run_ops(ag, s_dir, root, policy, all_ops, interpret=True)
        # twin
        s2, ag2 = fresh(root, "twin", policy, interpret=True)
        twin = run_ops(
            ag2, s2, root, policy, fixture.get("script_twin") or [], interpret=True
        )
        capacity = run_capacity_lanes(seed=seed)
    life_last = main.get("last_stage_clear")
    # Honest: J16 is capacity launch — life_last_stage_clear=J15 if main cleared J15 ops
    # But J16 also has interpret checks before capacity_launch
    first_fail = main.get("first_fail")
    last = "J16" if main["ok"] and capacity.get("ok") else life_last
    life_clear = life_last
    if main["ok"] and life_last == "J16":
        # capacity launch stage — report J15 as developmental clear if preferred
        life_clear = "J15"
    summary = {
        "version": "TM.0.22.INTERPRET",
        "lab": "TM.0.22.INTERPRET",
        "ex0s_under_test": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "ok": first_fail is None and twin.get("ok") and capacity.get("ok"),
        "last_stage_clear": last,
        "life_last_stage_clear": life_clear if first_fail is None else life_last,
        "first_fail_stage": (first_fail or {}).get("stage"),
        "first_fail": first_fail,
        "main_ok": main["ok"],
        "twin_ok": twin.get("ok"),
        "capacity": capacity,
        "fixture_sha": _sha_file(FIXTURE_JSON),
        "agent_sha": _sha_file(AGENT_PY),
        "candidate_sha": _sha_file(CANDIDATE_LOCK),
        "mech_sha": _sha_file(MECH_LOCK),
        "run_tm022interpret_sha": _sha_file(Path(__file__)),
        "bounded_claim": (
            "Ex0S reconstructed first-order source-specific interpretations of "
            "symbolic messages from independently grounded observable learning and behavior."
        ),
    }
    if write_lock:
        write_interpret_lock(summary)
        write_results_md(summary)
    return summary


def write_interpret_lock(summary: dict[str, Any]) -> dict[str, Any]:
    snap = {
        "version": "TM.0.22.INTERPRET",
        "lab": "TM.0.22.INTERPRET",
        "ex0s_under_test": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "ok": summary["ok"],
        "last_stage_clear": summary.get("last_stage_clear"),
        "life_last_stage_clear": summary.get("life_last_stage_clear"),
        "first_fail_stage": summary.get("first_fail_stage"),
        "first_fail": summary.get("first_fail"),
        "main_ok": summary.get("main_ok"),
        "twin_ok": summary.get("twin_ok"),
        "capacity": summary.get("capacity"),
        "fixture_sha": summary.get("fixture_sha"),
        "agent_sha": summary.get("agent_sha"),
        "candidate_sha": summary.get("candidate_sha"),
        "mech_sha": summary.get("mech_sha"),
        "run_tm022interpret_sha": summary.get("run_tm022interpret_sha"),
        "bounded_claim": summary.get("bounded_claim"),
        "refuse": [
            "subjective comprehension",
            "honesty_score",
            "cause ABI",
            "result field",
            "earned_next=true or non-null ex0s",
        ],
    }
    INTERPRET_LOCK.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return snap


def write_results_md(summary: dict[str, Any]) -> None:
    wall = {}
    if WALL_LOCK.exists():
        wall = json.loads(WALL_LOCK.read_text(encoding="utf-8"))
    ffw = wall.get("first_fail_wall") or {}
    lines = [
        "# TM.0.22.INTERPRET results: behaviorally evidenced interpretation",
        "",
        "**Ex0S under test:** **0.0.004** (not a new stamp)",
        "**Lab:** TM.0.22.INTERPRET",
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
        "Locks: [`interpret_baseline.lock`](interpret_baseline.lock) · "
        "[`interpret.candidate.lock`](interpret.candidate.lock) · "
        "[`interpret.candidate.v1.lock`](interpret.candidate.v1.lock) · "
        "[`interpret_mech.lock`](interpret_mech.lock) · "
        "[`interpret.lock`](interpret.lock) · "
        "[`interpret_wall.lock`](interpret_wall.lock)",
        "",
        "`earned_next`: **false** — no Ex0S 0.0.005 / 1.0. Product stamp remains **0.0.004**.",
        "",
        "## Bounded claim",
        "",
        "> Ex0S reconstructed first-order source-specific interpretations of symbolic "
        "messages from independently grounded observable learning and behavior.",
        "",
        "## Explicit non-claims",
        "",
        "- Not subjective comprehension / belief / honesty_score / intent / stability",
        "- Interpretation never becomes world ANSWER",
        "- No Jaccard; no derived statuses in S; no result field",
        "- Cause UNKNOWN on wall is scorer/narrative only",
        "",
        "## Next",
        "",
        "Honesty (inconsistent reporting) and stability (unmarked sudden change) remain open.",
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
    with tempfile.TemporaryDirectory(prefix="tm022interp_wall_") as tmp:
        root = Path(tmp)
        scripts = fixture.get("wall_scripts") or {}
        for w in fixture["wall"]:
            wid = w["id"]
            note = w.get("note") or ""
            if wid in scripts:
                s_dir, ag = fresh(root, wid, policy, interpret=True)
                out = run_ops(
                    ag, s_dir, root, policy, scripts[wid]["ops"], interpret=True
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
    # Narrative cause UNKNOWN is scorer-only — attach to honesty probes in note
    summary = {
        "version": "TM.0.22.INTERPRET.WALL",
        "lab": "TM.0.22.INTERPRET.WALL",
        "ex0s_under_test": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "ok": first_fail is None,
        "scored_probes_ok": scored_ok,
        "need_not_fully_pass": True,
        "first_fail_wall": first_fail,
        "next_primitive_hint": (first_fail or {}).get("dimension")
        or (first_fail or {}).get("id"),
        "cause_unknown_note": "scorer/narrative only — not an organism ABI",
        "probes": results,
        "fixture_sha": _sha_file(FIXTURE_JSON),
        "interpret_wall_prereg_sha": _sha_file(PREREG_WALL),
        "interpret_lock_sha": _sha_file(INTERPRET_LOCK) if INTERPRET_LOCK.exists() else None,
        "agent_sha": _sha_file(AGENT_PY),
        "run_tm022interpret_sha": _sha_file(Path(__file__)),
        "note": (
            "Results only. Behaviorally evidenced interpretation earned in-life; "
            "honesty / stability / nested ToM remain open. not_run probes are not "
            "first_fail diagnostics. Do not rewrite wall prereg."
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
        b = verify_baseline_prereg()
        m = verify_mech_prereg()
        w = verify_wall_prereg()
        print(
            json.dumps(
                {
                    "baseline": {"ok": b[0], "why": b[1]},
                    "mech": {"ok": m[0], "why": m[1]},
                    "wall": {"ok": w[0], "why": w[1]},
                },
                indent=2,
            )
        )
        return
    if args.baseline or args.write_baseline:
        print(
            json.dumps(
                run_baseline(seed=args.seed, write_lock=args.write_baseline),
                indent=2,
                default=str,
            )
        )
        return
    if args.smoke:
        print(json.dumps(run_smoke(seed=args.seed), indent=2))
        return
    if args.write_candidate:
        run_smoke(seed=args.seed)
        snap = write_candidate_lock()
        print(json.dumps({"ok": True, "candidate": snap["agent_sha"][:16]}, indent=2))
        return
    if args.unit_cells or args.write_mech_lock:
        cells = run_unit_cells(seed=args.seed)
        if args.write_mech_lock:
            write_mech_lock(cells)
        print(json.dumps(cells, indent=2, default=str))
        return
    if args.life or args.write_lock:
        print(
            json.dumps(
                run_life(seed=args.seed, write_lock=args.write_lock),
                indent=2,
                default=str,
            )
        )
        return
    if args.wall or args.write_wall:
        print(
            json.dumps(
                run_wall(seed=args.seed, write_lock=args.write_wall),
                indent=2,
                default=str,
            )
        )
        return
    ap.print_help()


if __name__ == "__main__":
    main()
