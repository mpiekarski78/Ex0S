"""TM.0.17.SYMBOLWORLD: grounded symbolic learning package.

Phases: A baseline wall (PERSIST-on) → B grounding candidate → C developmental life.
Product stays 0.0.004; earned_next=false; ex0s=null. No pixels/LLM/dictionary.
Runner generates nothing — only replays docs/symbol_world_fixture.json.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm014acquire import traverse_hold
from experiments.run_tm016persist import make_persist
from experiments.run_tm016relate import (
    clear_by_source,
    empty_birth,
    reload_store,
)
from three_memory import agent as agent_mod
from three_memory.policy import UsePolicy

PREREG_WORLD = REPO_ROOT / "docs" / "symbol_world.prereg.lock"
PREREG_GROUND = REPO_ROOT / "docs" / "symbol_ground.prereg.lock"
FIXTURE_JSON = REPO_ROOT / "docs" / "symbol_world_fixture.json"
BASELINE_LOCK = REPO_ROOT / "docs" / "symbol_world_baseline.lock"
CANDIDATE_LOCK = REPO_ROOT / "docs" / "symbol_ground.candidate.lock"
GROUND_LOCK = REPO_ROOT / "docs" / "symbol_ground.lock"
WORLD_LOCK = REPO_ROOT / "docs" / "symbol_world.lock"
BASELINE_MD = REPO_ROOT / "docs" / "tm017symbolworld_baseline_results.md"
RESULTS_MD = REPO_ROOT / "docs" / "tm017symbolworld_results.md"
PERSIST_LOCK = REPO_ROOT / "docs" / "persist.lock"
LIFE_WALL_LOCK = REPO_ROOT / "docs" / "life_wall.lock"
AGENT_PY = REPO_ROOT / "three_memory" / "agent.py"

DEFAULT_SEED = 12345
SOURCE_GROUND = "experience_grounding"
SOURCE_FP = "experience_fingerprint"
SOURCE_CONT = "experience_continuity"
MIN_SUPPORT = 2

WORLD_CLAIM = (
    "Can the same frozen mechanism learn words, actions, properties and simple "
    "composition from interaction inside an unfamiliar symbolic world? Phase A "
    "freezes the world and measures baseline absence: PERSIST-on must HOLD on "
    "unknown words. No LLM, dictionary, semantic labels, pixels or audio."
)

GROUND_CLAIM = (
    "An opt-in recipe may author raw co-occurrence rows into experience_grounding "
    "from exact observe_symbol_ground tuples and, at use time only, recompute "
    "evidence-weighted bindings between utterance symbols and paired world tokens "
    "so a unique winner among offered choices can be selected. The same machinery "
    "serves nouns, verbs, properties and roles — no POS-specific learners. Ties and "
    "insufficient support produce HOLD. Meaning is never stored as a synonym map."
)

STAGES = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10")


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _sha_src(fn: Callable[..., Any]) -> str:
    return _sha_bytes(inspect.getsource(fn).encode())


def load_fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_JSON.read_text(encoding="utf-8"))


def make_symbol_ground(s_dir: Path, policy: UsePolicy | None = None, **kwargs: Any) -> Any:
    ag = make_persist(s_dir, policy, **kwargs)
    ag.use_symbol_ground = True
    return ag


def fresh(tmp: Path, name: str, policy: UsePolicy, *, grounded: bool) -> tuple[Path, Any]:
    s_dir = tmp / name
    empty_birth(s_dir)
    ag = make_symbol_ground(s_dir, policy) if grounded else make_persist(s_dir, policy)
    ag.reset_rho()
    return s_dir, ag


def verify_world_prereg() -> tuple[bool, str, dict[str, Any]]:
    if not PREREG_WORLD.exists():
        return False, "docs/symbol_world.prereg.lock missing", {}
    lock = json.loads(PREREG_WORLD.read_text(encoding="utf-8"))
    if lock.get("lab") != "TM.0.17.SYMBOLWORLD":
        return False, "lab drift", lock
    if lock.get("earned_next") is not False or lock.get("ex0s") is not None:
        return False, "earn/product drift", lock
    if lock.get("preregistered_claim") != WORLD_CLAIM:
        return False, "claim drift", lock
    if lock.get("fixture_sha") != _sha_file(FIXTURE_JSON):
        return False, "fixture_sha pin", lock
    if lock.get("phase_a", {}).get("agent_edits_permitted") is not False:
        return False, "phase A must forbid agent edits", lock
    pins = lock.get("prior_lock_shas") or {}
    paths = {
        "persist.lock": PERSIST_LOCK,
        "life_wall.lock": LIFE_WALL_LOCK,
        "alias_finger.lock": REPO_ROOT / "docs" / "alias_finger.lock",
        "gap_wall.lock": REPO_ROOT / "docs" / "gap_wall.lock",
    }
    for name, path in paths.items():
        if pins.get(name) != _sha_file(path):
            return False, f"prior pin drift: {name}", lock
    return True, "symbol_world.prereg.lock intact", lock


def verify_ground_prereg() -> tuple[bool, str, dict[str, Any]]:
    if not PREREG_GROUND.exists():
        return False, "docs/symbol_ground.prereg.lock missing", {}
    lock = json.loads(PREREG_GROUND.read_text(encoding="utf-8"))
    if lock.get("lab") != "TM.0.17.SYMBOLWORLD.GROUND":
        return False, "lab drift", lock
    if lock.get("earned_next") is not False or lock.get("ex0s") is not None:
        return False, "earn/product drift", lock
    if lock.get("preregistered_claim") != GROUND_CLAIM:
        return False, "claim drift", lock
    if lock.get("flag") != "use_symbol_ground" or lock.get("flag_default") is not False:
        return False, "flag contract", lock
    if lock.get("source") != SOURCE_GROUND:
        return False, "source contract", lock
    banned = ("agent_sha", "run_tm017symbolworld_sha", "make_symbol_ground_sha")
    if any(k in lock for k in banned):
        return False, "prereg contains runner/agent SHAs", lock
    return True, "symbol_ground.prereg.lock intact", lock


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


def score_probe(ag: Any, probe: dict[str, Any], *, min_support: int = MIN_SUPPORT) -> tuple[bool, Any]:
    expect = probe["expect"]
    expression = probe.get("evidence_family") == "expression"
    sel = ag.select_grounded(
        probe["utterance"],
        probe["choices"],
        min_support=min_support,
        expression=expression,
    )
    actual = sel.get("selected")
    if expect == "HOLD":
        return actual is None, actual
    return str(actual or "").lower() == str(expect).lower(), actual


def score_traverse(ag: Any, cue: str, expect: str) -> tuple[bool, Any]:
    trav = traverse_hold(ag, cue)
    lived = trav.get("lived_bind")
    if expect == "HOLD":
        return lived is None, lived
    return str(lived or "").lower() == str(expect).lower(), lived


def seed_isolation_rows(ag: Any) -> None:
    """Plant FP + continuity rows that must not substitute for word meaning."""
    # Fingerprint noise on the utterance token — not a grounding binding.
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
            raise RuntimeError(f"isolation alias_probe reject: {out}")
    # Continuity noise tying ball/cup spellings — must not become select_grounded.
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
            raise RuntimeError(f"isolation continuity reject: {out}")


def run_fork(
    tmp: Path,
    s_dir: Path,
    policy: UsePolicy,
    fork: dict[str, Any],
) -> dict[str, Any] | None:
    kind = fork["kind"]
    clone = tmp / f"fork_{kind}_{fork['stage']}_{fork.get('probes', [{}])[0].get('id', 'x')}"
    if clone.exists():
        shutil.rmtree(clone)
    shutil.copytree(s_dir, clone)
    ag = make_symbol_ground(clone, policy)
    reload_store(ag)
    ag.reset_rho()
    if kind == "strip_grounding":
        clear_by_source(clone, SOURCE_GROUND)
        reload_store(ag)
        ag.reset_rho()
    elif kind == "reset_rho":
        ag.reset_rho()
    elif kind == "strip_fingerprint":
        clear_by_source(clone, SOURCE_FP)
        reload_store(ag)
        ag.reset_rho()
    elif kind == "strip_continuity":
        clear_by_source(clone, SOURCE_CONT)
        reload_store(ag)
        ag.reset_rho()
    elif kind == "donor_swap":
        clear_by_source(clone, SOURCE_GROUND)
        reload_store(ag)
        for probe in fork["probes"]:
            for row in probe.get("donor_rows") or []:
                apply_ground(ag, row)
        ag.reset_rho()
    else:
        raise ValueError(kind)
    for probe in fork["probes"]:
        ok, actual = score_probe(ag, probe)
        if not ok:
            return {
                "stage": fork["stage"],
                "lane": kind,
                "probe": probe.get("id"),
                "expected": probe["expect"],
                "actual": actual,
                "evidence_family": probe.get("evidence_family") or "grounding",
            }
    return None


def run_script(
    tmp: Path,
    name: str,
    policy: UsePolicy,
    script: Sequence[dict[str, Any]],
    *,
    grounded: bool,
    lane: str,
    min_support: int = MIN_SUPPORT,
) -> dict[str, Any]:
    s_dir, ag = fresh(tmp, name, policy, grounded=grounded)
    last_clear = None
    first_fail: dict[str, Any] | None = None
    n_probe = 0

    for op in script:
        kind = op["op"]
        if kind == "event":
            ag.observe_event({"visible": list(op["visible"]), "focus": None})
            if op.get("end_episode"):
                ag.end_event_episode()
            continue
        if kind == "end_episode":
            ag.end_event_episode()
            continue
        if kind == "reset_rho":
            ag.reset_rho()
            continue
        if kind == "stage_marker":
            continue
        if kind == "isolation_seed":
            if grounded:
                seed_isolation_rows(ag)
            continue
        if kind == "trial":
            if not grounded:
                # Baseline: scene visibles only; never write grounding.
                ag.observe_event({"visible": list(op.get("visibles") or []), "focus": None})
                ag.end_event_episode()
                continue
            for row in op["ground_rows"]:
                apply_ground(ag, row)
            ag.observe_event({"visible": list(op.get("visibles") or []), "focus": None})
            ag.end_event_episode()
            continue
        if kind == "ground":
            if grounded:
                apply_ground(ag, op)
            continue
        if kind == "checkpoint_traverse":
            n_probe += 1
            if first_fail is not None:
                continue
            ok, actual = score_traverse(ag, op["cue"], op["expect"])
            if not ok:
                first_fail = {
                    "stage": op.get("stage"),
                    "lane": lane,
                    "probe": op["id"],
                    "expected": op["expect"],
                    "actual": actual,
                    "evidence_family": "traverse",
                }
            continue
        if kind == "probe":
            n_probe += 1
            if first_fail is not None:
                continue
            if not grounded:
                # Baseline wall: always expect HOLD regardless of fixture expect
                # for developmental probes that leaked into baseline — baseline
                # script already sets HOLD.
                ok, actual = score_probe(ag, op, min_support=min_support)
            else:
                ok, actual = score_probe(ag, op, min_support=min_support)
            if not ok:
                first_fail = {
                    "stage": op.get("stage"),
                    "lane": lane,
                    "probe": op["id"],
                    "expected": op["expect"],
                    "actual": actual,
                    "evidence_family": op.get("evidence_family") or "grounding",
                }
            else:
                st = op.get("stage")
                if st in STAGES:
                    if last_clear is None or STAGES.index(st) >= STAGES.index(last_clear):
                        last_clear = st
            continue
        if kind == "fork":
            if first_fail is not None or not grounded:
                continue
            fail = run_fork(tmp, s_dir, policy, op)
            if fail is not None:
                first_fail = fail
            continue
        raise ValueError(f"unknown op {kind}")

    return {
        "lane": lane,
        "ok": first_fail is None,
        "first_fail": first_fail,
        "last_stage_clear": last_clear,
        "n_probes": n_probe,
    }


def run_baseline(*, seed: int = DEFAULT_SEED, write_lock: bool = False) -> dict[str, Any]:
    ok_p, why_p, _ = verify_world_prereg()
    if not ok_p:
        raise RuntimeError(why_p)
    fixture = load_fixture()
    policy = UsePolicy(seed=seed)
    with tempfile.TemporaryDirectory(prefix="tm017sw_base_") as tmp:
        result = run_script(
            Path(tmp),
            "baseline",
            policy,
            fixture["script_baseline"],
            grounded=False,
            lane="baseline",
        )
    summary = {
        "version": "TM.0.17.SYMBOLWORLD.BASELINE",
        "lab": "TM.0.17.SYMBOLWORLD",
        "phase": "A",
        "ok": result["ok"],
        "earned_next": False,
        "ex0s": None,
        "claim": "PERSIST-on does not magically ground words",
        "first_fail": result["first_fail"],
        "n_probes": result["n_probes"],
        "fixture_sha": _sha_file(FIXTURE_JSON),
        "agent_sha": _sha_file(AGENT_PY),
        "factory": "make_persist",
    }
    if write_lock:
        write_baseline_lock(summary)
        BASELINE_MD.write_text(
            "\n".join(
                [
                    "# TM.0.17.SYMBOLWORLD Phase A — baseline wall",
                    "",
                    f"**Recorded:** {'PASS' if summary['ok'] else 'FAIL'} — PERSIST-on unknown-word HOLD",
                    "",
                    "- Product: `0.0.004`",
                    "- `earned_next=false`",
                    "- `ex0s=null`",
                    "- Factory: `make_persist` (no grounding)",
                    "",
                    "## Next",
                    "",
                    "Phase B: one general grounding candidate (`use_symbol_ground`).",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    return summary


def write_baseline_lock(summary: dict[str, Any]) -> dict[str, Any]:
    snap = {
        "version": "TM.0.17.SYMBOLWORLD.BASELINE",
        "lab": "TM.0.17.SYMBOLWORLD",
        "phase": "A",
        "ex0s_under_test": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "ok": summary["ok"],
        "first_fail": summary["first_fail"],
        "fixture_sha": _sha_file(FIXTURE_JSON),
        "symbol_world_prereg_sha": _sha_file(PREREG_WORLD),
        "agent_sha": _sha_file(AGENT_PY),
        "persist_agent_sha": json.loads(PERSIST_LOCK.read_text())["agent_sha"],
        "run_tm017symbolworld_sha": _sha_file(Path(__file__)),
        "refuse": [
            "editing agent.py in Phase A",
            "run-time curriculum generation",
            "pixels / LLM / dictionary",
            "earned_next=true or non-null ex0s",
        ],
    }
    BASELINE_LOCK.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return snap


def write_candidate_lock() -> dict[str, Any]:
    snap = {
        "version": "TM.0.17.SYMBOLWORLD.GROUND.CANDIDATE",
        "lab": "TM.0.17.SYMBOLWORLD.GROUND",
        "ex0s_under_test": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "flag": "use_symbol_ground",
        "source": SOURCE_GROUND,
        "observation_abi": "observe_symbol_ground",
        "factory": "experiments.run_tm017symbolworld.make_symbol_ground",
        "agent_sha": _sha_file(AGENT_PY),
        "observe_symbol_ground_sha": _sha_src(agent_mod.ThreeMemoryAgent.observe_symbol_ground),
        "select_grounded_sha": _sha_src(agent_mod.ThreeMemoryAgent.select_grounded),
        "make_symbol_ground_sha": _sha_src(make_symbol_ground),
        "run_tm017symbolworld_sha": _sha_file(Path(__file__)),
        "symbol_ground_prereg_sha": _sha_file(PREREG_GROUND),
        "symbol_world_baseline_sha": _sha_file(BASELINE_LOCK) if BASELINE_LOCK.exists() else None,
        "prior_persist_agent_sha": json.loads(PERSIST_LOCK.read_text())["agent_sha"],
        "note": "Pinned before unit-cell / developmental scoring. Do not rewrite after score.",
    }
    CANDIDATE_LOCK.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return snap


def run_unit_cells(*, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    ok_p, why_p, _ = verify_ground_prereg()
    if not ok_p:
        raise RuntimeError(why_p)
    if not CANDIDATE_LOCK.exists():
        raise RuntimeError("candidate.lock missing — write before scoring")
    fixture = load_fixture()
    policy = UsePolicy(seed=seed)
    cells_out = []
    with tempfile.TemporaryDirectory(prefix="tm017sw_cells_") as tmp:
        root = Path(tmp)
        for name, cell in fixture["unit_cells"].items():
            s_dir = root / name
            empty_birth(s_dir)
            if cell.get("flag"):
                ag = make_symbol_ground(s_dir, policy)
            else:
                ag = make_persist(s_dir, policy)
            ag.reset_rho()
            why = "ok"
            ok = True
            if cell.get("malformed") is not None:
                bad = ag.observe_symbol_ground(cell["malformed"])
                ok = bad.get("ok") is False and bad.get("why") == "exact_key_reject"
                why = bad.get("why")
            for row in cell.get("grounds") or []:
                if cell.get("flag"):
                    apply_ground(ag, row)
                else:
                    off = ag.observe_symbol_ground(row) if hasattr(ag, "observe_symbol_ground") else {"why": "missing"}
                    if off.get("ok"):
                        ok = False
                        why = "flag_off_wrote"
            if cell.get("strip"):
                clear_by_source(s_dir, SOURCE_GROUND)
                reload_store(ag)
                ag.reset_rho()
            if cell.get("donor_swap_to"):
                clear_by_source(s_dir, SOURCE_GROUND)
                reload_store(ag)
                for row in cell["donor_swap_to"]:
                    apply_ground(ag, row)
            poke = cell["probe"]
            if cell.get("flag") is False:
                # Flag off: select_grounded reports grounding_off / None
                sel = (
                    ag.select_grounded(poke["utterance"], poke["choices"])
                    if hasattr(ag, "select_grounded")
                    else {"selected": None, "why": "missing"}
                )
                actual = sel.get("selected")
                cell_ok = actual is None
            else:
                cell_ok, actual = score_probe(ag, poke)
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


def run_life(*, seed: int = DEFAULT_SEED, write_lock: bool = False) -> dict[str, Any]:
    ok_p, why_p, _ = verify_world_prereg()
    if not ok_p:
        raise RuntimeError(why_p)
    ok_g, why_g, _ = verify_ground_prereg()
    if not ok_g:
        raise RuntimeError(why_g)
    if not CANDIDATE_LOCK.exists():
        raise RuntimeError("candidate.lock required before developmental score")
    fixture = load_fixture()
    policy = UsePolicy(seed=seed)
    with tempfile.TemporaryDirectory(prefix="tm017sw_life_") as tmp:
        root = Path(tmp)
        main = run_script(
            root, "main", policy, fixture["script_life"], grounded=True, lane="main"
        )
        twin = run_script(
            root, "twin", policy, fixture["script_twin"], grounded=True, lane="twin"
        )
    fails = [r["first_fail"] for r in (main, twin) if r.get("first_fail")]
    first_fail = fails[0] if fails else None
    if first_fail is None:
        last = "S10"
        first_stage = None
    else:
        first_stage = first_fail.get("stage")
        # last clear is previous stage in STAGES
        if first_stage in STAGES:
            idx = STAGES.index(first_stage)
            last = STAGES[idx - 1] if idx > 0 else None
        else:
            last = main.get("last_stage_clear")
    summary = {
        "version": "TM.0.17.SYMBOLWORLD",
        "lab": "TM.0.17.SYMBOLWORLD",
        "ok": first_fail is None,
        "earned_next": False,
        "ex0s": None,
        "claim": WORLD_CLAIM,
        "last_stage_clear": last,
        "first_fail_stage": first_stage,
        "first_fail": first_fail,
        "main": {k: v for k, v in main.items() if k != "first_fail" or v},
        "twin": {k: v for k, v in twin.items() if k != "first_fail" or v},
        "fixture_sha": _sha_file(FIXTURE_JSON),
        "agent_sha": _sha_file(AGENT_PY),
        "candidate_sha": _sha_file(CANDIDATE_LOCK),
    }
    if write_lock:
        write_world_lock(summary)
        write_results_md(summary)
    return summary


def write_ground_lock(cells: dict[str, Any]) -> dict[str, Any]:
    snap = {
        "version": "TM.0.17.SYMBOLWORLD.GROUND",
        "lab": "TM.0.17.SYMBOLWORLD.GROUND",
        "ex0s_under_test": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "ok": cells["ok"],
        "n_pass": cells["n_pass"],
        "n_cells": cells["n_cells"],
        "cells": cells["cells"],
        "agent_sha": _sha_file(AGENT_PY),
        "candidate_sha": _sha_file(CANDIDATE_LOCK),
        "run_tm017symbolworld_sha": _sha_file(Path(__file__)),
    }
    GROUND_LOCK.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return snap


def write_world_lock(summary: dict[str, Any]) -> dict[str, Any]:
    snap = {
        "version": "TM.0.17.SYMBOLWORLD",
        "lab": "TM.0.17.SYMBOLWORLD",
        "ex0s_under_test": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "last_stage_clear": summary["last_stage_clear"],
        "first_fail_stage": summary["first_fail_stage"],
        "first_fail": summary["first_fail"],
        "main_ok": summary["main"]["ok"],
        "twin_ok": summary["twin"]["ok"],
        "fixture_sha": _sha_file(FIXTURE_JSON),
        "symbol_world_prereg_sha": _sha_file(PREREG_WORLD),
        "symbol_ground_prereg_sha": _sha_file(PREREG_GROUND),
        "candidate_sha": _sha_file(CANDIDATE_LOCK),
        "agent_sha": _sha_file(AGENT_PY),
        "run_tm017symbolworld_sha": _sha_file(Path(__file__)),
        "refuse": [
            "mechanism changes between stages",
            "POS-specific learners",
            "pixels / LLM / dictionary",
            "free sentence generation",
            "earned_next=true or non-null ex0s / Ex0S 1.0",
            "rewriting symbol_world_fixture.json after pin",
        ],
    }
    WORLD_LOCK.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return snap


def write_results_md(summary: dict[str, Any]) -> None:
    ff = summary.get("first_fail")
    if ff is None:
        wall = f"Cleared through **{summary['last_stage_clear']}** (main + twin)."
    else:
        wall = (
            f"Last clear **{summary['last_stage_clear']}**; first fail "
            f"**{summary['first_fail_stage']}**: "
            f"`({ff.get('stage')}, {ff.get('lane')}, {ff.get('probe')}, "
            f"{ff.get('expected')}, {ff.get('actual')}, {ff.get('evidence_family')})`."
        )
    RESULTS_MD.write_text(
        "\n".join(
            [
                "# TM.0.17.SYMBOLWORLD results",
                "",
                f"**Recorded:** developmental life → **{'PASS' if summary['ok'] else 'WALL'}**",
                "",
                "- Product: `0.0.004`",
                "- `earned_next=false`",
                "- `ex0s=null`",
                "- Mechanism: `use_symbol_ground` / `experience_grounding`",
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
                "## Bounded fact",
                "",
                "One general evidence-weighted grounding substrate in an unfamiliar "
                "symbolic world. Alias fingerprints and continuity marks stay isolated. "
                "Not a product stamp.",
                "",
                "## Next",
                "",
                "Discuss capability naming only if the wall is fully green; otherwise "
                "diagnose first_fail_stage. No Ex0S 1.0.",
                "",
                "## Reproduce",
                "",
                "```bash",
                "python -m experiments.run_tm017symbolworld --verify-prereg",
                "python tests/test_tm017symbolworld.py",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-prereg", action="store_true")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--write-baseline", action="store_true")
    ap.add_argument("--write-candidate", action="store_true")
    ap.add_argument("--unit-cells", action="store_true")
    ap.add_argument("--write-ground-lock", action="store_true")
    ap.add_argument("--life", action="store_true")
    ap.add_argument("--write-lock", action="store_true")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    if args.verify_prereg:
        results = {}
        for name, fn in (
            ("world", verify_world_prereg),
            ("ground", verify_ground_prereg),
        ):
            if name == "ground" and not PREREG_GROUND.exists():
                results[name] = {"ok": True, "why": "not yet authored"}
                continue
            ok, why, _ = fn()
            results[name] = {"ok": ok, "why": why}
        print(json.dumps(results, indent=2))
        sys.exit(0 if all(v["ok"] for v in results.values()) else 1)

    if args.write_candidate:
        if not BASELINE_LOCK.exists():
            raise SystemExit("baseline lock required before candidate")
        snap = write_candidate_lock()
        print(json.dumps({"ok": True, "candidate_sha": _sha_file(CANDIDATE_LOCK), **{k: snap[k] for k in ("agent_sha", "flag")}}, indent=2))
        return

    if args.baseline or args.write_baseline:
        summary = run_baseline(seed=args.seed, write_lock=args.write_baseline)
        print(json.dumps(summary, indent=2))
        sys.exit(0 if summary["ok"] else 1)

    if args.unit_cells or args.write_ground_lock:
        cells = run_unit_cells(seed=args.seed)
        if args.write_ground_lock:
            if not cells["ok"]:
                raise SystemExit("unit cells not earned — not freezing ground lock")
            write_ground_lock(cells)
        print(json.dumps(cells, indent=2))
        sys.exit(0 if cells["ok"] else 1)

    if args.life or args.write_lock:
        summary = run_life(seed=args.seed, write_lock=args.write_lock)
        out = {k: v for k, v in summary.items() if k not in ("main", "twin")}
        out["main_ok"] = summary["main"]["ok"]
        out["twin_ok"] = summary["twin"]["ok"]
        print(json.dumps(out, indent=2))
        sys.exit(0)

    ap.print_help()


if __name__ == "__main__":
    main()
