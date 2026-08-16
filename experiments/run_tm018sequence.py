"""TM.0.18.SEQUENCE: grounded symbolic expression package.

Phases: A baseline (SYMBOLWORLD-on) → B sequence candidate → C expressive life
→ unconfounded capacity lanes → preregistered dialogue wall.
Product stays 0.0.004; earned_next=false; ex0s=null.
Runner replays docs/sequence_fixture.json — no run-time curriculum invention.
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
from experiments.run_tm017symbolworld import make_symbol_ground
from three_memory import agent as agent_mod
from three_memory.policy import UsePolicy

PREREG_BASELINE = REPO_ROOT / "docs" / "sequence_baseline.prereg.lock"
PREREG_MECH = REPO_ROOT / "docs" / "sequence_mech.prereg.lock"
PREREG_DIALOGUE = REPO_ROOT / "docs" / "sequence_dialogue.prereg.lock"
FIXTURE_JSON = REPO_ROOT / "docs" / "sequence_fixture.json"
BASELINE_LOCK = REPO_ROOT / "docs" / "sequence_baseline.lock"
CANDIDATE_LOCK = REPO_ROOT / "docs" / "sequence.candidate.lock"
MECH_LOCK = REPO_ROOT / "docs" / "sequence_mech.lock"
SEQUENCE_LOCK = REPO_ROOT / "docs" / "sequence.lock"
DIALOGUE_LOCK = REPO_ROOT / "docs" / "sequence_dialogue.lock"
RESULTS_MD = REPO_ROOT / "docs" / "tm018sequence_results.md"
CONTRACT_MD = REPO_ROOT / "docs" / "sequence_evidence_contract.md"
AGENT_PY = REPO_ROOT / "three_memory" / "agent.py"
SYMBOL_WORLD_LOCK = REPO_ROOT / "docs" / "symbol_world.lock"
SYMBOL_GROUND_LOCK = REPO_ROOT / "docs" / "symbol_ground.lock"
PERSIST_LOCK = REPO_ROOT / "docs" / "persist.lock"

DEFAULT_SEED = 12345
SOURCE_SEQ = "experience_sequence"
SOURCE_GROUND = "experience_grounding"
SOURCE_FP = "experience_fingerprint"
SOURCE_CONT = "experience_continuity"
MIN_SUPPORT = 2
EMIT_CAP = 64
STAGES = (
    "E0",
    "E1",
    "E2",
    "E3",
    "E4",
    "E5",
    "E6",
    "E7",
    "E8",
    "E9",
    "E10",
    "E11",
    "E12",
)

BASELINE_CLAIM = (
    "Frozen SYMBOLWORLD (make_symbol_ground) can select a bounded learned word but "
    "cannot construct an unseen variable-length utterance; unknown or ambiguous "
    "construction requests remain HOLD. No sequence mechanism in Phase A."
)

MECH_CLAIM = (
    "An opt-in recipe may author raw sequence-step rows into experience_sequence from "
    "exact observe_sequence_step tuples over factorized context atoms, input symbols, "
    "and output prefix, and at use time recompute unique next_operation emit|stop "
    "(with next_symbol on emit) so emit_sequence constructs a variable-length utterance "
    "or returns atomic HOLD. No scene IDs, grammar slots, menus, or complete-response "
    "lookup. STOP placement is evidenced. Cap=64 does not reveal expected length."
)

DIALOGUE_CLAIM = (
    "On frozen make_sequence, a preregistered dialogue wall probes delayed correction, "
    "earlier-event reference, two-relation questions, interruption/topic return, "
    "contradiction, and unknown HOLD. Need not fully pass; first_fail_dialogue diagnoses "
    "the next missing primitive. Fixture frozen before candidate score."
)


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _sha_src(fn: Callable[..., Any]) -> str:
    return _sha_bytes(inspect.getsource(fn).encode())


def load_fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_JSON.read_text(encoding="utf-8"))


def make_sequence(s_dir: Path, policy: UsePolicy | None = None, **kwargs: Any) -> Any:
    ag = make_symbol_ground(s_dir, policy, **kwargs)
    ag.use_symbol_sequence = True
    return ag


def fresh(tmp: Path, name: str, policy: UsePolicy, *, sequenced: bool) -> tuple[Path, Any]:
    s_dir = tmp / name
    empty_birth(s_dir)
    ag = make_sequence(s_dir, policy) if sequenced else make_symbol_ground(s_dir, policy)
    ag.reset_rho()
    return s_dir, ag


def verify_baseline_prereg() -> tuple[bool, str, dict[str, Any]]:
    if not PREREG_BASELINE.exists():
        return False, "docs/sequence_baseline.prereg.lock missing", {}
    lock = json.loads(PREREG_BASELINE.read_text(encoding="utf-8"))
    if lock.get("lab") != "TM.0.18.SEQUENCE":
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
    for name, path in (
        ("symbol_world.lock", SYMBOL_WORLD_LOCK),
        ("symbol_ground.lock", SYMBOL_GROUND_LOCK),
        ("persist.lock", PERSIST_LOCK),
    ):
        if pins.get(name) != _sha_file(path):
            return False, f"prior pin drift: {name}", lock
    return True, "sequence_baseline.prereg.lock intact", lock


def verify_mech_prereg() -> tuple[bool, str, dict[str, Any]]:
    if not PREREG_MECH.exists():
        return False, "docs/sequence_mech.prereg.lock missing", {}
    lock = json.loads(PREREG_MECH.read_text(encoding="utf-8"))
    if lock.get("lab") != "TM.0.18.SEQUENCE.MECH":
        return False, "lab drift", lock
    if lock.get("earned_next") is not False or lock.get("ex0s") is not None:
        return False, "earn/product drift", lock
    if lock.get("preregistered_claim") != MECH_CLAIM:
        return False, "claim drift", lock
    if lock.get("flag") != "use_symbol_sequence" or lock.get("flag_default") is not False:
        return False, "flag contract", lock
    if lock.get("source") != SOURCE_SEQ:
        return False, "source contract", lock
    if lock.get("emit_cap") != EMIT_CAP:
        return False, "emit_cap", lock
    banned = ("agent_sha", "run_tm018sequence_sha", "make_sequence_sha")
    if any(k in lock for k in banned):
        return False, "prereg contains runner/agent SHAs", lock
    return True, "sequence_mech.prereg.lock intact", lock


def verify_dialogue_prereg() -> tuple[bool, str, dict[str, Any]]:
    if not PREREG_DIALOGUE.exists():
        return False, "docs/sequence_dialogue.prereg.lock missing", {}
    lock = json.loads(PREREG_DIALOGUE.read_text(encoding="utf-8"))
    if lock.get("lab") != "TM.0.18.SEQUENCE.DIALOGUE":
        return False, "lab drift", lock
    if lock.get("earned_next") is not False or lock.get("ex0s") is not None:
        return False, "earn/product drift", lock
    if lock.get("preregistered_claim") != DIALOGUE_CLAIM:
        return False, "claim drift", lock
    if lock.get("fixture_sha") != _sha_file(FIXTURE_JSON):
        return False, "fixture_sha pin", lock
    if lock.get("mechanism_changes_permitted") is not False:
        return False, "dialogue must freeze mechanism", lock
    fix = load_fixture()
    if lock.get("probe_ids") != [d["id"] for d in fix["dialogue"]]:
        return False, "dialogue probe_ids drift", lock
    return True, "sequence_dialogue.prereg.lock intact", lock


def apply_ground(ag: Any, row: dict[str, Any]) -> None:
    out = ag.observe_symbol_ground(
        {
            "symbol": row["symbol"],
            "paired": row["paired"],
            "trial_id": row["trial_id"],
            "result": row["result"],
        }
    )
    if not out.get("ok"):
        raise RuntimeError(f"ground reject: {out}")


def apply_seq_step(ag: Any, step: dict[str, Any]) -> None:
    payload = {
        "context_atoms": list(step["context_atoms"]),
        "input_symbols": list(step["input_symbols"]),
        "prefix": list(step.get("prefix") or []),
        "next_operation": step["next_operation"],
        "next_symbol": step.get("next_symbol") or "",
        "result": step.get("result") or "success",
    }
    out = ag.observe_sequence_step(payload)
    if not out.get("ok"):
        raise RuntimeError(f"seq_step reject: {out}")


def score_select(ag: Any, probe: dict[str, Any], *, min_support: int = MIN_SUPPORT) -> tuple[bool, Any]:
    expect = probe["expect"]
    sel = ag.select_grounded(
        probe["utterance"],
        probe["choices"],
        min_support=min_support,
        expression=bool(probe.get("expression")),
    )
    actual = sel.get("selected")
    if expect == "HOLD":
        return actual is None, actual
    return str(actual or "").lower() == str(expect).lower(), actual


def score_emit(
    ag: Any,
    probe: dict[str, Any],
    *,
    min_support: int = MIN_SUPPORT,
    cap: int = EMIT_CAP,
    timings: list[float] | None = None,
) -> tuple[bool, Any]:
    t0 = time.perf_counter()
    if not hasattr(ag, "emit_sequence"):
        actual = None
        why = "missing_emit"
        ok = probe.get("expect") == "HOLD"
        if timings is not None:
            timings.append(time.perf_counter() - t0)
        return ok, actual
    res = ag.emit_sequence(
        probe["context_atoms"],
        probe["input_symbols"],
        min_support=min_support,
        cap=cap,
    )
    dt = time.perf_counter() - t0
    if timings is not None:
        timings.append(dt)
    actual = res.get("sequence")
    if probe.get("expect") == "HOLD":
        return actual is None, actual
    expect_seq = [str(x).lower() for x in probe.get("expect_seq") or []]
    if actual is None:
        return False, None
    return [str(x).lower() for x in actual] == expect_seq, actual


def seed_fp_cont(ag: Any) -> None:
    for ctx in ("ctx_alpha", "ctx_beta"):
        out = ag.observe_alias_probe(
            {
                "alias": "ball",
                "probe_context": ctx,
                "action": "press",
                "observed_outcome": "success",
            }
        )
        if not out.get("ok"):
            raise RuntimeError(f"fp seed reject: {out}")
    for phase, op, token, state in (
        ("pre_gap", "apply", "ball", "on"),
        ("post_gap", "read", "cup", "on"),
    ):
        out = ag.observe_continuity_mark(
            {
                "token": token,
                "mark_id": "iso_mk",
                "phase": phase,
                "operation": op,
                "observed_state": state,
            }
        )
        if not out.get("ok"):
            raise RuntimeError(f"cont seed reject: {out}")


def teach_equal_order(ag: Any, teach: dict[str, Any]) -> None:
    ctx = list(teach["context"])
    inp = ["describe"]
    n = int(teach.get("n") or 2)
    for seq in (teach["seq_a"], teach["seq_b"]):
        for _ in range(n):
            pref: list[str] = []
            for tok in seq:
                apply_seq_step(
                    ag,
                    {
                        "context_atoms": ctx,
                        "input_symbols": inp,
                        "prefix": list(pref),
                        "next_operation": "emit",
                        "next_symbol": tok,
                        "result": "success",
                    },
                )
                pref.append(tok)
            apply_seq_step(
                ag,
                {
                    "context_atoms": ctx,
                    "input_symbols": inp,
                    "prefix": list(pref),
                    "next_operation": "stop",
                    "next_symbol": "",
                    "result": "success",
                },
            )
    # Attest each context atom and align seq tokens position-wise for the gate.
    for i, atom in enumerate(ctx):
        apply_ground(
            ag,
            {
                "symbol": f"eqatom_{i}",
                "paired": atom,
                "trial_id": f"eq_atom_{i}_0",
                "result": "success",
            },
        )
        apply_ground(
            ag,
            {
                "symbol": f"eqatom_{i}",
                "paired": atom,
                "trial_id": f"eq_atom_{i}_1",
                "result": "success",
            },
        )
    for seq in (teach["seq_a"], teach["seq_b"]):
        for j, tok in enumerate(seq):
            atom = ctx[j] if j < len(ctx) else ctx[-1]
            apply_ground(
                ag,
                {
                    "symbol": tok,
                    "paired": atom,
                    "trial_id": f"eq_sym_{tok}_{j}_0",
                    "result": "success",
                },
            )
            apply_ground(
                ag,
                {
                    "symbol": tok,
                    "paired": atom,
                    "trial_id": f"eq_sym_{tok}_{j}_1",
                    "result": "success",
                },
            )


def run_fork(
    tmp: Path,
    s_dir: Path,
    policy: UsePolicy,
    fork: dict[str, Any],
) -> dict[str, Any] | None:
    kind = fork["kind"]
    clone = tmp / f"fork_{kind}_{fork.get('stage')}_{fork.get('id', 'x')}"
    if clone.exists():
        shutil.rmtree(clone)
    shutil.copytree(s_dir, clone)
    ag = make_sequence(clone, policy)
    reload_store(ag)
    ag.reset_rho()
    if kind == "strip_sequence":
        clear_by_source(clone, SOURCE_SEQ)
        reload_store(ag)
        ag.reset_rho()
    elif kind == "strip_grounding":
        clear_by_source(clone, SOURCE_GROUND)
        reload_store(ag)
        ag.reset_rho()
    elif kind == "fp_cont_only":
        clear_by_source(clone, SOURCE_SEQ)
        clear_by_source(clone, SOURCE_GROUND)
        reload_store(ag)
        seed_fp_cont(ag)
        ag.reset_rho()
    elif kind == "donor_sequence":
        clear_by_source(clone, SOURCE_SEQ)
        reload_store(ag)
        for step in fork.get("donor_steps") or []:
            apply_seq_step(ag, step)
        ag.reset_rho()
    else:
        raise ValueError(kind)
    for probe in fork["probes"]:
        ok, actual = score_emit(ag, probe)
        if not ok:
            return {
                "stage": fork.get("stage"),
                "lane": kind,
                "probe": probe.get("id"),
                "expected": probe.get("expect") or probe.get("expect_seq"),
                "actual": actual,
                "failure_family": probe.get("failure_family") or "isolation",
            }
    return None


def run_fork_world(
    tmp: Path,
    s_dir: Path,
    policy: UsePolicy,
    fork: dict[str, Any],
) -> dict[str, Any] | None:
    kind = fork["kind"]
    clone = tmp / f"forkworld_{kind}_{fork.get('id', 'x')}"
    if clone.exists():
        shutil.rmtree(clone)
    shutil.copytree(s_dir, clone)
    ag = make_sequence(clone, policy)
    reload_store(ag)
    ag.reset_rho()
    if kind == "equal_order_hold":
        for teach in fork.get("teach") or []:
            teach_equal_order(ag, teach)
    else:
        raise ValueError(kind)
    ok, actual = score_emit(ag, fork["probe"])
    if not ok:
        return {
            "stage": fork.get("stage"),
            "lane": kind,
            "probe": fork.get("id"),
            "expected": fork["probe"].get("expect") or fork["probe"].get("expect_seq"),
            "actual": actual,
            "failure_family": "ordering",
        }
    return None


def run_script(
    tmp: Path,
    name: str,
    policy: UsePolicy,
    script: Sequence[dict[str, Any]],
    *,
    sequenced: bool,
    lane: str,
    min_support: int = MIN_SUPPORT,
    collect_timings: bool = False,
) -> dict[str, Any]:
    s_dir, ag = fresh(tmp, name, policy, sequenced=sequenced)
    last_clear = None
    first_fail: dict[str, Any] | None = None
    n_probe = 0
    timings: list[float] = []
    rows_examined: list[int] = []

    for op in script:
        kind = op["op"]
        if kind == "event":
            ag.observe_event({"visible": list(op["visible"]), "focus": None})
            if op.get("end_episode"):
                ag.end_event_episode()
            continue
        if kind == "reset_rho":
            ag.reset_rho()
            continue
        if kind == "stage_marker":
            continue
        if kind == "ground_batch":
            for row in op["rows"]:
                apply_ground(ag, row)
            continue
        if kind == "seq_step":
            if sequenced:
                apply_seq_step(ag, op)
            continue
        if kind == "probe_select":
            n_probe += 1
            if first_fail is not None:
                continue
            ok, actual = score_select(ag, op, min_support=min_support)
            if not ok:
                first_fail = {
                    "stage": op.get("stage"),
                    "lane": lane,
                    "probe": op.get("id"),
                    "expected": op.get("expect"),
                    "actual": actual,
                    "failure_family": op.get("failure_family") or "grounding",
                }
            else:
                st = op.get("stage")
                if st in STAGES:
                    if last_clear is None or STAGES.index(st) >= STAGES.index(last_clear):
                        last_clear = st
            continue
        if kind == "probe_emit":
            n_probe += 1
            if first_fail is not None:
                continue
            n_rows = len(ag.store.records()) if hasattr(ag, "store") else 0
            rows_examined.append(n_rows)
            ok, actual = score_emit(
                ag,
                op,
                min_support=min_support,
                timings=timings if collect_timings else None,
            )
            if not ok:
                fif = getattr(ag, "first_internal_fail", None)
                first_fail = {
                    "stage": op.get("stage"),
                    "lane": lane,
                    "probe": op.get("id"),
                    "expected": op.get("expect") or op.get("expect_seq"),
                    "actual": actual,
                    "failure_family": op.get("failure_family") or "unknown",
                    "first_internal_fail": fif,
                }
            else:
                st = op.get("stage")
                if st in STAGES:
                    if last_clear is None or STAGES.index(st) >= STAGES.index(last_clear):
                        last_clear = st
            continue
        if kind == "fork":
            if first_fail is not None or not sequenced:
                continue
            fail = run_fork(tmp, s_dir, policy, op)
            if fail is not None:
                first_fail = fail
            continue
        if kind == "fork_world":
            if first_fail is not None or not sequenced:
                continue
            fail = run_fork_world(tmp, s_dir, policy, op)
            if fail is not None:
                first_fail = fail
            continue
        raise ValueError(f"unknown op {kind}")

    metrics: dict[str, Any] = {
        "s_row_count": len(ag.store.records()) if hasattr(ag, "store") else 0,
        "n_emit_timings": len(timings),
    }
    if timings:
        metrics["p50_emit_s"] = float(statistics.median(timings))
        metrics["p95_emit_s"] = float(
            sorted(timings)[max(0, int(round(0.95 * (len(timings) - 1))))]
        )
        metrics["complete_utterance_latency_s"] = float(sum(timings) / len(timings))
    if rows_examined:
        metrics["evidence_rows_examined_p50"] = float(statistics.median(rows_examined))

    return {
        "lane": lane,
        "ok": first_fail is None,
        "first_fail": first_fail,
        "last_stage_clear": last_clear,
        "n_probes": n_probe,
        "metrics": metrics,
        "s_dir": str(s_dir),
        "ag": ag,
    }


def run_baseline(*, seed: int = DEFAULT_SEED, write_lock: bool = False) -> dict[str, Any]:
    ok_p, why_p, _ = verify_baseline_prereg()
    if not ok_p:
        raise RuntimeError(why_p)
    fixture = load_fixture()
    policy = UsePolicy(seed=seed)
    with tempfile.TemporaryDirectory(prefix="tm018seq_base_") as tmp:
        result = run_script(
            Path(tmp),
            "baseline",
            policy,
            fixture["script_baseline"],
            sequenced=False,
            lane="baseline",
        )
    summary = {
        "version": "TM.0.18.SEQUENCE.BASELINE",
        "lab": "TM.0.18.SEQUENCE",
        "phase": "A",
        "ok": result["ok"],
        "earned_next": False,
        "ex0s": None,
        "claim": BASELINE_CLAIM,
        "first_fail": result["first_fail"],
        "n_probes": result["n_probes"],
        "fixture_sha": _sha_file(FIXTURE_JSON),
        "agent_sha": _sha_file(AGENT_PY),
        "factory": "make_symbol_ground",
    }
    if write_lock:
        snap = {
            "version": "TM.0.18.SEQUENCE.BASELINE",
            "lab": "TM.0.18.SEQUENCE",
            "phase": "A",
            "ex0s_under_test": "0.0.004",
            "earned_next": False,
            "ex0s": None,
            "ok": summary["ok"],
            "first_fail": summary["first_fail"],
            "fixture_sha": _sha_file(FIXTURE_JSON),
            "sequence_baseline_prereg_sha": _sha_file(PREREG_BASELINE),
            "agent_sha": _sha_file(AGENT_PY),
            "run_tm018sequence_sha": _sha_file(Path(__file__)),
            "refuse": [
                "editing agent.py in Phase A",
                "scene IDs / answer hashes",
                "run-time curriculum generation",
                "earned_next=true or non-null ex0s",
            ],
        }
        BASELINE_LOCK.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return summary


def write_candidate_lock() -> dict[str, Any]:
    snap = {
        "version": "TM.0.18.SEQUENCE.CANDIDATE",
        "lab": "TM.0.18.SEQUENCE.MECH",
        "ex0s_under_test": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "flag": "use_symbol_sequence",
        "source": SOURCE_SEQ,
        "observation_abi": "observe_sequence_step",
        "emission_abi": "emit_sequence",
        "factory": "experiments.run_tm018sequence.make_sequence",
        "agent_sha": _sha_file(AGENT_PY),
        "observe_sequence_step_sha": _sha_src(agent_mod.ThreeMemoryAgent.observe_sequence_step),
        "emit_sequence_sha": _sha_src(agent_mod.ThreeMemoryAgent.emit_sequence),
        "make_sequence_sha": _sha_src(make_sequence),
        "run_tm018sequence_sha": _sha_file(Path(__file__)),
        "sequence_mech_prereg_sha": _sha_file(PREREG_MECH),
        "sequence_baseline_sha": _sha_file(BASELINE_LOCK) if BASELINE_LOCK.exists() else None,
        "sequence_dialogue_prereg_sha": _sha_file(PREREG_DIALOGUE),
        "note": "Pinned after unscored ABI smoke, before scored cells / E-life / capacity / dialogue.",
    }
    CANDIDATE_LOCK.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return snap


def run_smoke(*, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Unscored ABI smoke — must precede candidate.lock."""
    ok_p, why_p, _ = verify_mech_prereg()
    if not ok_p:
        raise RuntimeError(why_p)
    policy = UsePolicy(seed=seed)
    with tempfile.TemporaryDirectory(prefix="tm018seq_smoke_") as tmp:
        s_dir, ag = fresh(Path(tmp), "smoke", policy, sequenced=True)
        bad = ag.observe_sequence_step({"context_atoms": ["a"], "next_operation": "emit"})
        if bad.get("ok") or bad.get("why") != "exact_key_reject":
            return {"ok": False, "why": f"reject smoke failed: {bad}"}
        apply_ground(
            ag,
            {"symbol": "w", "paired": "c", "trial_id": "s0", "result": "success"},
        )
        apply_ground(
            ag,
            {"symbol": "w", "paired": "c", "trial_id": "s1", "result": "success"},
        )
        for _ in range(2):
            apply_seq_step(
                ag,
                {
                    "context_atoms": ["c"],
                    "input_symbols": ["q"],
                    "prefix": [],
                    "next_operation": "emit",
                    "next_symbol": "w",
                    "result": "success",
                },
            )
            apply_seq_step(
                ag,
                {
                    "context_atoms": ["c"],
                    "input_symbols": ["q"],
                    "prefix": ["w"],
                    "next_operation": "stop",
                    "next_symbol": "",
                    "result": "success",
                },
            )
        # Unscored: just ensure emit returns something or HOLD without crashing
        _ = ag.emit_sequence(["c"], ["q"], min_support=MIN_SUPPORT, cap=EMIT_CAP)
    return {"ok": True, "why": "abi_smoke"}


def run_unit_cells(*, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    ok_p, why_p, _ = verify_mech_prereg()
    if not ok_p:
        raise RuntimeError(why_p)
    if not CANDIDATE_LOCK.exists():
        raise RuntimeError("candidate.lock missing — write before scoring")
    fixture = load_fixture()
    policy = UsePolicy(seed=seed)
    cells_out = []
    with tempfile.TemporaryDirectory(prefix="tm018seq_cells_") as tmp:
        root = Path(tmp)
        for name, cell in fixture["unit_cells"].items():
            s_dir = root / name
            empty_birth(s_dir)
            if cell.get("flag"):
                ag = make_sequence(s_dir, policy)
            else:
                ag = make_symbol_ground(s_dir, policy)
            ag.reset_rho()
            why = "ok"
            ok = True
            if cell.get("malformed") is not None:
                bad = ag.observe_sequence_step(cell["malformed"])
                ok = bad.get("ok") is False and bad.get("why") == "exact_key_reject"
                why = bad.get("why")
            for row in cell.get("grounds") or []:
                if cell.get("flag") and hasattr(ag, "observe_symbol_ground"):
                    apply_ground(ag, row)
            for step in cell.get("steps") or []:
                if cell.get("flag"):
                    apply_seq_step(ag, step)
                else:
                    off = (
                        ag.observe_sequence_step(
                            {
                                "context_atoms": step["context_atoms"],
                                "input_symbols": step["input_symbols"],
                                "prefix": step.get("prefix") or [],
                                "next_operation": step["next_operation"],
                                "next_symbol": step.get("next_symbol") or "",
                                "result": step.get("result") or "success",
                            }
                        )
                        if hasattr(ag, "observe_sequence_step")
                        else {"why": "missing"}
                    )
                    if off.get("ok"):
                        ok = False
                        why = "flag_off_wrote"
            if cell.get("strip_sequence"):
                clear_by_source(s_dir, SOURCE_SEQ)
                reload_store(ag)
                ag.reset_rho()
            if cell.get("donor_steps"):
                clear_by_source(s_dir, SOURCE_SEQ)
                reload_store(ag)
                for step in cell["donor_steps"]:
                    apply_seq_step(ag, step)
                ag.reset_rho()
            poke = cell["probe"]
            if cell.get("flag") is False:
                res = (
                    ag.emit_sequence(poke["context_atoms"], poke["input_symbols"])
                    if hasattr(ag, "emit_sequence")
                    else {"sequence": None, "why": "missing"}
                )
                actual = res.get("sequence")
                cell_ok = actual is None
            else:
                cell_ok, actual = score_emit(ag, poke)
            cells_out.append(
                {
                    "cell": name,
                    "ok": bool(ok and cell_ok),
                    "actual": actual,
                    "why": why if not ok else ("pass" if cell_ok else f"got {actual}"),
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
        "version": "TM.0.18.SEQUENCE.MECH",
        "lab": "TM.0.18.SEQUENCE.MECH",
        "ex0s_under_test": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "ok": cells["ok"],
        "n_pass": cells["n_pass"],
        "n_cells": cells["n_cells"],
        "cells": cells["cells"],
        "agent_sha": _sha_file(AGENT_PY),
        "candidate_sha": _sha_file(CANDIDATE_LOCK),
        "run_tm018sequence_sha": _sha_file(Path(__file__)),
    }
    MECH_LOCK.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return snap


def run_capacity_lanes(*, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    fixture = load_fixture()
    policy = UsePolicy(seed=seed)
    lanes_out: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="tm018seq_cap_") as tmp:
        root = Path(tmp)
        for lane_name, lane in fixture["capacity"].items():
            first_fail_rung = None
            rung_results = []
            for branch in lane.get("branches") or []:
                rung = branch["rung"]
                result = run_script(
                    root,
                    f"{lane_name}_{rung}",
                    policy,
                    branch["script"],
                    sequenced=True,
                    lane=lane_name,
                    collect_timings=True,
                )
                # drop live agent from serialization
                result.pop("ag", None)
                result.pop("s_dir", None)
                rung_results.append(
                    {
                        "rung": rung,
                        "ok": result["ok"],
                        "first_fail": result["first_fail"],
                        "metrics": result["metrics"],
                    }
                )
                if not result["ok"] and first_fail_rung is None:
                    first_fail_rung = rung
            lanes_out[lane_name] = {
                "ok": first_fail_rung is None,
                "first_fail_rung": first_fail_rung,
                "rungs": rung_results,
            }
    return {"ok": all(v["ok"] for v in lanes_out.values()), "lanes": lanes_out}


def run_life(*, seed: int = DEFAULT_SEED, write_lock: bool = False) -> dict[str, Any]:
    ok_b, why_b, _ = verify_baseline_prereg()
    if not ok_b:
        raise RuntimeError(why_b)
    ok_m, why_m, _ = verify_mech_prereg()
    if not ok_m:
        raise RuntimeError(why_m)
    ok_d, why_d, _ = verify_dialogue_prereg()
    if not ok_d:
        raise RuntimeError(why_d)
    if not CANDIDATE_LOCK.exists():
        raise RuntimeError("candidate.lock required before life score")
    if not MECH_LOCK.exists():
        raise RuntimeError("sequence_mech.lock required before life score")
    fixture = load_fixture()
    policy = UsePolicy(seed=seed)
    with tempfile.TemporaryDirectory(prefix="tm018seq_life_") as tmp:
        root = Path(tmp)
        main = run_script(
            root,
            "main",
            policy,
            fixture["script_life"],
            sequenced=True,
            lane="main",
            collect_timings=True,
        )
        twin = run_script(
            root,
            "twin",
            policy,
            fixture["script_twin"],
            sequenced=True,
            lane="twin",
            collect_timings=True,
        )
        capacity = run_capacity_lanes(seed=seed)
    for r in (main, twin):
        r.pop("ag", None)
        r.pop("s_dir", None)
    fails = [r["first_fail"] for r in (main, twin) if r.get("first_fail")]
    first_fail = fails[0] if fails else None
    life_last = main.get("last_stage_clear")
    if first_fail is None and main["ok"] and twin["ok"]:
        # E12 is the capacity-launch stage (no life probe). Package clears E12
        # only when unconfounded capacity lanes also pass; life_last stays honest.
        first_stage = None
        if capacity.get("ok"):
            last = "E12"
        else:
            last = life_last
    else:
        first_stage = first_fail.get("stage") if first_fail else None
        if first_stage in STAGES:
            idx = STAGES.index(first_stage)
            last = STAGES[idx - 1] if idx > 0 else None
        else:
            last = life_last
    # operational failure_family only when fork isolated
    if first_fail and first_fail.get("failure_family") not in {
        "grounding",
        "ordering",
        "retrieval",
        "generation",
        "capacity",
        "isolation",
        "unknown",
    }:
        first_fail["failure_family"] = "unknown"
    # Non-fork probe labels in the fixture are hints only — force unknown unless isolation/ordering.
    if first_fail and first_fail.get("lane") not in {
        "strip_sequence",
        "strip_grounding",
        "fp_cont_only",
        "donor_sequence",
        "equal_order_hold",
    }:
        if first_fail.get("failure_family") == "isolation":
            first_fail["failure_family"] = "unknown"
    summary = {
        "version": "TM.0.18.SEQUENCE",
        "lab": "TM.0.18.SEQUENCE",
        "ok": first_fail is None and capacity.get("ok", False),
        "earned_next": False,
        "ex0s": None,
        "claim": (
            "A frozen, generic mechanism learned termination and variable-length ordered "
            "construction from factorized grounded evidence, composed an unseen utterance, "
            "followed a counterfactually reordered language, and remained causally dependent "
            "on grounding and sequence evidence in S."
        ),
        "last_stage_clear": last,
        "life_last_stage_clear": life_last,
        "first_fail_stage": first_stage,
        "first_fail": first_fail,
        "first_internal_fail": (first_fail or {}).get("first_internal_fail"),
        "failure_family": (first_fail or {}).get("failure_family"),
        "main": {k: v for k, v in main.items() if k not in ("ag", "s_dir")},
        "twin": {k: v for k, v in twin.items() if k not in ("ag", "s_dir")},
        "capacity": capacity,
        "fixture_sha": _sha_file(FIXTURE_JSON),
        "agent_sha": _sha_file(AGENT_PY),
        "candidate_sha": _sha_file(CANDIDATE_LOCK),
    }
    if write_lock:
        write_sequence_lock(summary)
        write_results_md(summary)
    return summary


def write_sequence_lock(summary: dict[str, Any]) -> dict[str, Any]:
    snap = {
        "version": "TM.0.18.SEQUENCE",
        "lab": "TM.0.18.SEQUENCE",
        "ex0s_under_test": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "ok": summary["ok"],
        "last_stage_clear": summary["last_stage_clear"],
        "life_last_stage_clear": summary.get("life_last_stage_clear"),
        "first_fail_stage": summary["first_fail_stage"],
        "first_fail": summary["first_fail"],
        "first_internal_fail": summary.get("first_internal_fail"),
        "failure_family": summary.get("failure_family"),
        "main_ok": summary["main"]["ok"],
        "twin_ok": summary["twin"]["ok"],
        "capacity": summary.get("capacity"),
        "wall_metrics": {
            "main": summary["main"].get("metrics"),
            "twin": summary["twin"].get("metrics"),
        },
        "fixture_sha": _sha_file(FIXTURE_JSON),
        "sequence_baseline_prereg_sha": _sha_file(PREREG_BASELINE),
        "sequence_mech_prereg_sha": _sha_file(PREREG_MECH),
        "sequence_dialogue_prereg_sha": _sha_file(PREREG_DIALOGUE),
        "candidate_sha": _sha_file(CANDIDATE_LOCK),
        "agent_sha": _sha_file(AGENT_PY),
        "run_tm018sequence_sha": _sha_file(Path(__file__)),
        "refuse": [
            "mechanism changes between stages",
            "scene IDs / grammar slots / menus-as-speech",
            "partial-credit prefixes",
            "earned_next=true or non-null ex0s / Ex0S 1.0",
            "rewriting sequence_dialogue.prereg.lock after score",
        ],
    }
    SEQUENCE_LOCK.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return snap


def write_results_md(summary: dict[str, Any]) -> None:
    ff = summary.get("first_fail")
    if ff is None and summary.get("ok"):
                wall = f"Cleared through **{summary['last_stage_clear']}** (main + twin + capacity)."
                if summary.get("life_last_stage_clear"):
                    wall += (
                        f" Life probes last cleared **{summary['life_last_stage_clear']}**; "
                        "E12 is the capacity-launch stage."
                    )
    elif ff is None:
        wall = (
            f"Life cleared through **{summary['last_stage_clear']}**; capacity wall "
            f"recorded under capacity lanes."
        )
    else:
        wall = (
            f"Last clear **{summary['last_stage_clear']}**; first fail "
            f"**{summary['first_fail_stage']}**: "
            f"`({ff.get('stage')}, {ff.get('lane')}, {ff.get('probe')}, "
            f"{ff.get('expected')}, {ff.get('actual')}, {ff.get('failure_family')})`."
        )
    cap = summary.get("capacity") or {}
    cap_lines = []
    for lane, data in (cap.get("lanes") or {}).items():
        cap_lines.append(
            f"| {lane} | {data.get('ok')} | {data.get('first_fail_rung')} |"
        )
    RESULTS_MD.write_text(
        "\n".join(
            [
                "# TM.0.18.SEQUENCE results",
                "",
                f"**Recorded:** expressive life → **{'PASS' if summary['ok'] else 'WALL'}**",
                "",
                "- Product: `0.0.004`",
                "- `earned_next=false`",
                "- `ex0s=null`",
                "- Mechanism: `use_symbol_sequence` / `experience_sequence`",
                "",
                "## Capacity",
                "",
                wall,
                "",
                "| Lane | ok | last_stage_clear | probes |",
                "|------|----|------------------|--------|",
                f"| main | {summary['main']['ok']} | {summary['main'].get('last_stage_clear')} | {summary['main']['n_probes']} |",
                f"| twin | {summary['twin']['ok']} | {summary['twin'].get('last_stage_clear')} | {summary['twin']['n_probes']} |",
                "",
                "## Unconfounded capacity lanes",
                "",
                "| Lane | ok | first_fail_rung |",
                "|------|----|-----------------|",
                *(cap_lines or ["| (none) | | |"]),
                "",
                "## Wall metrics",
                "",
                f"- main: `{json.dumps(summary['main'].get('metrics'))}`",
                f"- twin: `{json.dumps(summary['twin'].get('metrics'))}`",
                "",
                "## Bounded fact",
                "",
                summary.get("claim") or "",
                "",
                "## Next",
                "",
                "Dialogue wall is diagnostic only. No Ex0S 1.0 / product stamp.",
                "",
                "## Reproduce",
                "",
                "```bash",
                "python -m experiments.run_tm018sequence --verify-prereg",
                "python tests/test_tm018sequence.py",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_dialogue(*, seed: int = DEFAULT_SEED, write_lock: bool = False) -> dict[str, Any]:
    ok_d, why_d, _ = verify_dialogue_prereg()
    if not ok_d:
        raise RuntimeError(why_d)
    if not SEQUENCE_LOCK.exists():
        raise RuntimeError("sequence.lock required before dialogue wall")
    fixture = load_fixture()
    policy = UsePolicy(seed=seed)
    results = []
    first_fail = None
    with tempfile.TemporaryDirectory(prefix="tm018seq_dlg_") as tmp:
        root = Path(tmp)
        # Rebuild a short world from life script then apply dialogue probes.
        life = run_script(
            root,
            "dlg_life",
            policy,
            fixture["script_life"],
            sequenced=True,
            lane="dialogue_prep",
        )
        ag = life.pop("ag")
        s_dir = Path(life.pop("s_dir"))
        seed = fixture.get("dialogue_seed") or {}
        for probe in fixture["dialogue"]:
            pid = probe["id"]
            if pid in seed:
                # fresh clone for contradict seed
                clone = root / f"dlg_{pid}"
                if clone.exists():
                    shutil.rmtree(clone)
                empty_birth(clone)
                ag_p = make_sequence(clone, policy)
                for row in seed[pid].get("grounds") or []:
                    apply_ground(ag_p, row)
                for step in seed[pid].get("steps") or []:
                    apply_seq_step(ag_p, step)
                target = ag_p
            else:
                target = ag
            ok, actual = score_emit(target, probe)
            if probe.get("followup") and ok:
                fu = dict(probe["followup"])
                fu["context_atoms"] = probe["context_atoms"]
                ok2, actual2 = score_emit(target, fu)
                ok = ok and ok2
                if not ok2:
                    actual = actual2
            row = {
                "id": pid,
                "ok": ok,
                "expected": probe.get("expect") or probe.get("expect_seq"),
                "actual": actual,
                "note": probe.get("note"),
            }
            results.append(row)
            if not ok and first_fail is None:
                first_fail = row
    summary = {
        "version": "TM.0.18.SEQUENCE.DIALOGUE",
        "lab": "TM.0.18.SEQUENCE.DIALOGUE",
        "ex0s_under_test": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "ok": first_fail is None,
        "need_not_fully_pass": True,
        "first_fail_dialogue": first_fail,
        "probes": results,
        "fixture_sha": _sha_file(FIXTURE_JSON),
        "sequence_dialogue_prereg_sha": _sha_file(PREREG_DIALOGUE),
        "sequence_lock_sha": _sha_file(SEQUENCE_LOCK),
        "agent_sha": _sha_file(AGENT_PY),
        "run_tm018sequence_sha": _sha_file(Path(__file__)),
        "note": "Results only. Do not rewrite dialogue prereg.",
    }
    if write_lock:
        DIALOGUE_LOCK.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
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
    ap.add_argument("--dialogue", action="store_true")
    ap.add_argument("--write-dialogue", action="store_true")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    if args.verify_prereg:
        results = {}
        for name, fn in (
            ("baseline", verify_baseline_prereg),
            ("mech", verify_mech_prereg),
            ("dialogue", verify_dialogue_prereg),
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
            raise SystemExit("baseline lock required before candidate")
        smoke = run_smoke(seed=args.seed)
        if not smoke["ok"]:
            raise SystemExit(f"smoke failed: {smoke}")
        snap = write_candidate_lock()
        print(
            json.dumps(
                {
                    "ok": True,
                    "candidate_sha": _sha_file(CANDIDATE_LOCK),
                    "agent_sha": snap["agent_sha"],
                    "flag": snap["flag"],
                },
                indent=2,
            )
        )
        return

    if args.baseline or args.write_baseline:
        summary = run_baseline(seed=args.seed, write_lock=args.write_baseline)
        print(json.dumps({k: v for k, v in summary.items() if k != "claim"}, indent=2))
        sys.exit(0 if summary["ok"] else 1)

    if args.unit_cells:
        cells = run_unit_cells(seed=args.seed)
        if args.write_mech_lock:
            if not cells["ok"]:
                raise SystemExit("cells not earned — refusing mech lock")
            write_mech_lock(cells)
        print(json.dumps(cells, indent=2))
        sys.exit(0 if cells["ok"] else 1)

    if args.life or args.write_lock:
        summary = run_life(seed=args.seed, write_lock=args.write_lock)
        print(
            json.dumps(
                {
                    "ok": summary["ok"],
                    "last_stage_clear": summary["last_stage_clear"],
                    "first_fail_stage": summary["first_fail_stage"],
                    "first_fail": summary["first_fail"],
                    "capacity_ok": summary.get("capacity", {}).get("ok"),
                },
                indent=2,
            )
        )
        sys.exit(0 if summary["main"]["ok"] and summary["twin"]["ok"] else 1)

    if args.dialogue or args.write_dialogue:
        summary = run_dialogue(seed=args.seed, write_lock=args.write_dialogue)
        print(
            json.dumps(
                {
                    "ok": summary["ok"],
                    "first_fail_dialogue": summary["first_fail_dialogue"],
                    "n_probes": len(summary["probes"]),
                },
                indent=2,
            )
        )
        return

    ap.print_help()


if __name__ == "__main__":
    main()
