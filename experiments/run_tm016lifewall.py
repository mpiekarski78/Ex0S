"""TM.0.16.LIFEWALL: continuous-lifetime integration wall on frozen PERSIST-on.

Wall probe only. Product stays 0.0.004; earned_next=false; ex0s=null.
No agent.py edits. Runner generates nothing — only replays life_wall_fixture.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm014acquire import traverse_hold
from experiments.run_tm016persist import make_persist
from experiments.run_tm016relate import (
    clear_by_source,
    empty_birth,
    reload_store,
)
from three_memory.policy import UsePolicy

PREREG_LOCK = REPO_ROOT / "docs" / "life_wall.prereg.lock"
FIXTURE_JSON = REPO_ROOT / "docs" / "life_wall_fixture.json"
LIFE_WALL_LOCK = REPO_ROOT / "docs" / "life_wall.lock"
RESULTS_MD = REPO_ROOT / "docs" / "tm016lifewall_results.md"
PERSIST_LOCK = REPO_ROOT / "docs" / "persist.lock"
AGENT_PY = REPO_ROOT / "three_memory" / "agent.py"

DEFAULT_SEED = 12345
SOURCE_CONT = "experience_continuity"
SOURCE_FP = "experience_fingerprint"

CLAIM = (
    "Measure whether frozen PERSIST-on (ALIASFINGER + mark-continuity) remains "
    "independently causal and behaviorally correct when RELATE, fingerprints, and "
    "continuity marks are interleaved throughout one accumulating lifetime (plus a "
    "causally remapped twin), scoring the trajectory at preregistered checkpoints "
    "and capacity rungs 4→8→16→32. This wall does not add machinery."
)


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def load_fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_JSON.read_text(encoding="utf-8"))


def load_prereg() -> dict[str, Any]:
    return json.loads(PREREG_LOCK.read_text(encoding="utf-8"))


def verify_prereg_lock() -> tuple[bool, str, dict[str, Any]]:
    if not PREREG_LOCK.exists():
        return False, "docs/life_wall.prereg.lock missing", {}
    lock = load_prereg()
    if lock.get("lab") != "TM.0.16.LIFEWALL":
        return False, "lab drift", lock
    if lock.get("earned_next") is not False:
        return False, "earned_next must be false", lock
    if lock.get("ex0s") is not None:
        return False, "ex0s must be null", lock
    if lock.get("preregistered_claim") != CLAIM:
        return False, "claim drift", lock
    if lock.get("organism", {}).get("agent_edits_permitted") is not False:
        return False, "agent edits must be forbidden", lock
    if lock.get("fixture_sha") != _sha_file(FIXTURE_JSON):
        return False, "fixture_sha pin", lock
    persist = json.loads(PERSIST_LOCK.read_text(encoding="utf-8"))
    # Historical agent pin from PERSIST earn; later labs may extend agent.py.
    if lock.get("frozen_agent_sha") != persist.get("agent_sha"):
        return False, "frozen_agent_sha != persist.lock agent_sha", lock
    if not lock.get("frozen_agent_sha"):
        return False, "frozen_agent_sha missing", lock
    pins = lock.get("prior_lock_shas") or {}
    paths = {
        "persist.lock": PERSIST_LOCK,
        "persist.prereg.lock": REPO_ROOT / "docs" / "persist.prereg.lock",
        "alias_finger.lock": REPO_ROOT / "docs" / "alias_finger.lock",
        "gap_wall.lock": REPO_ROOT / "docs" / "gap_wall.lock",
    }
    for name, path in paths.items():
        if pins.get(name) != _sha_file(path):
            return False, f"prior pin drift: {name}", lock
    banned = ("agent_sha", "run_tm016lifewall_sha", "make_persist_sha")
    if any(k in lock for k in banned):
        return False, "prereg contains runner/agent SHAs", lock
    return True, "life_wall.prereg.lock intact", lock


def fresh_world(tmp: Path, name: str, policy: UsePolicy) -> tuple[Path, Any]:
    s_dir = tmp / name
    empty_birth(s_dir)
    ag = make_persist(s_dir, policy)
    ag.reset_rho()
    return s_dir, ag


def _holds(trav: dict[str, Any]) -> bool:
    return trav.get("lived_bind") is None


def _query(ag: Any, cue: str) -> dict[str, Any]:
    ag.reset_rho()
    return traverse_hold(ag, cue)


def score_probe(ag: Any, cue: str, expect: str) -> tuple[bool, Any]:
    trav = _query(ag, cue)
    lived = trav.get("lived_bind")
    if expect == "HOLD":
        ok = lived is None
        return ok, lived
    ok = str(lived or "").lower() == str(expect).lower()
    return ok, lived


def apply_op(ag: Any, op: dict[str, Any]) -> None:
    kind = op["op"]
    if kind == "event":
        ag.observe_event({"visible": list(op["visible"]), "focus": None})
        if op.get("end_episode"):
            ag.end_event_episode()
    elif kind == "end_episode":
        ag.end_event_episode()
    elif kind == "continuity":
        out = ag.observe_continuity_mark(
            {
                "token": op["token"],
                "mark_id": op["mark_id"],
                "phase": op["phase"],
                "operation": op["operation"],
                "observed_state": op["observed_state"],
            }
        )
        if not out.get("ok"):
            raise RuntimeError(f"continuity reject: {out}")
    elif kind == "alias_probe":
        out = ag.observe_alias_probe(
            {
                "alias": op["alias"],
                "probe_context": op["probe_context"],
                "action": op["action"],
                "observed_outcome": op["observed_outcome"],
            }
        )
        if not out.get("ok"):
            raise RuntimeError(f"alias_probe reject: {out}")
    elif kind == "reset_rho":
        ag.reset_rho()
    else:
        raise ValueError(f"unknown op {kind}")


def clone_s(src: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)


def run_fork(
    tmp: Path,
    s_dir: Path,
    policy: UsePolicy,
    fork: dict[str, Any],
    *,
    lane_prefix: str,
) -> dict[str, Any] | None:
    """Return first_fail dict or None if all probes pass."""
    kind = fork["kind"]
    clone_path = tmp / f"fork_{lane_prefix}_{kind}_{fork['rung']}_{fork['phase']}"
    clone_s(s_dir, clone_path)
    ag = make_persist(clone_path, policy)
    reload_store(ag)
    ag.reset_rho()
    if kind == "reset_rho":
        ag.reset_rho()
    elif kind == "strip_continuity":
        clear_by_source(clone_path, SOURCE_CONT)
        reload_store(ag)
        ag.reset_rho()
    elif kind == "strip_fingerprint":
        clear_by_source(clone_path, SOURCE_FP)
        reload_store(ag)
        ag.reset_rho()
    else:
        raise ValueError(kind)
    lane = kind if kind != "reset_rho" else "reset_rho"
    for probe in fork["probes"]:
        ok, actual = score_probe(ag, probe["cue"], probe["expect"])
        if not ok:
            return {
                "rung": fork["rung"],
                "lane": lane,
                "checkpoint": probe.get("id") or probe["cue"],
                "object_index": probe.get("object_index"),
                "expected": probe["expect"],
                "actual": actual,
                "phase": fork["phase"],
            }
    return None


def run_script(
    tmp: Path,
    name: str,
    policy: UsePolicy,
    script: list[dict[str, Any]],
    *,
    lane: str,
) -> dict[str, Any]:
    s_dir, ag = fresh_world(tmp, name, policy)
    last_ok_rung = 0
    first_fail: dict[str, Any] | None = None
    n_ck = 0
    n_fork = 0

    for op in script:
        kind = op["op"]
        if kind in ("event", "end_episode", "continuity", "alias_probe", "reset_rho"):
            apply_op(ag, op)
            continue
        if kind == "checkpoint":
            n_ck += 1
            if first_fail is not None:
                continue
            ok, actual = score_probe(ag, op["cue"], op["expect"])
            if not ok:
                first_fail = {
                    "rung": op["rung"],
                    "lane": lane,
                    "checkpoint": op["id"],
                    "object_index": op.get("object_index"),
                    "expected": op["expect"],
                    "actual": actual,
                    "phase": op["phase"],
                }
            else:
                last_ok_rung = max(last_ok_rung, int(op["rung"]))
            continue
        if kind == "fork":
            n_fork += 1
            if first_fail is not None:
                continue
            fail = run_fork(tmp, s_dir, policy, op, lane_prefix=lane)
            if fail is not None:
                first_fail = fail
            continue
        raise ValueError(f"unknown op {kind}")

    return {
        "lane": lane,
        "ok": first_fail is None,
        "first_fail": first_fail,
        "last_ok_rung": (
            last_ok_rung
            if first_fail is None
            else (max([r for r in (4, 8, 16, 32) if r < int(first_fail["rung"])] or [0]))
        ),
        "n_checkpoints": n_ck,
        "n_forks": n_fork,
    }


def run_lifewall(*, seed: int = DEFAULT_SEED, write_lock: bool = False) -> dict[str, Any]:
    ok_p, why_p, _ = verify_prereg_lock()
    if not ok_p:
        raise RuntimeError(why_p)
    fixture = load_fixture()
    policy = UsePolicy(seed=seed)

    with tempfile.TemporaryDirectory(prefix="tm016lifewall_") as tmp:
        root = Path(tmp)
        main = run_script(root, "main", policy, fixture["script_main"], lane="main")
        twin = run_script(root, "twin", policy, fixture["script_twin"], lane="twin")

    fails = [r["first_fail"] for r in (main, twin) if r.get("first_fail")]
    first_fail = fails[0] if fails else None
    # last_ok_rung: highest rung fully cleared on both lanes
    if first_fail is None:
        last_ok = 32
        first_fail_rung = None
    else:
        first_fail_rung = int(first_fail["rung"])
        # A rung clears only when main+twin+forks all pass; failure at rung N means last_ok is previous
        cleared = [r for r in (4, 8, 16, 32) if r < first_fail_rung]
        last_ok = cleared[-1] if cleared else 0

    summary: dict[str, Any] = {
        "version": "TM.0.16.LIFEWALL",
        "lab": "TM.0.16.LIFEWALL",
        "label": "continuous-lifetime integration wall",
        "ok": first_fail is None,
        "earned_next": False,
        "ex0s": None,
        "seed": seed,
        "claim": CLAIM,
        "last_ok_rung": last_ok,
        "first_fail_rung": first_fail_rung,
        "first_fail": first_fail,
        "main": {k: v for k, v in main.items() if k != "first_fail" or v},
        "twin": {k: v for k, v in twin.items() if k != "first_fail" or v},
        "fixture_sha": _sha_file(FIXTURE_JSON),
        "agent_sha": _sha_file(AGENT_PY),
    }
    if write_lock:
        write_life_wall_lock(summary)
        write_results_md(summary)
    return summary


def write_life_wall_lock(summary: dict[str, Any], path: Path = LIFE_WALL_LOCK) -> dict[str, Any]:
    snap = {
        "version": "TM.0.16.LIFEWALL",
        "lab": "TM.0.16.LIFEWALL",
        "ex0s_under_test": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "not_tm017": True,
        "label": "continuous-lifetime integration wall on frozen PERSIST-on",
        "last_ok_rung": summary["last_ok_rung"],
        "first_fail_rung": summary["first_fail_rung"],
        "first_fail": summary["first_fail"],
        "main_ok": summary["main"]["ok"],
        "twin_ok": summary["twin"]["ok"],
        "fixture_sha": _sha_file(FIXTURE_JSON),
        "life_wall_prereg_sha": _sha_file(PREREG_LOCK),
        "agent_sha": _sha_file(AGENT_PY),
        "run_tm016lifewall_sha": _sha_file(Path(__file__)),
        "prior_lock_shas": {
            "persist.lock": _sha_file(PERSIST_LOCK),
            "alias_finger.lock": _sha_file(REPO_ROOT / "docs" / "alias_finger.lock"),
            "gap_wall.lock": _sha_file(REPO_ROOT / "docs" / "gap_wall.lock"),
        },
        "refuse": [
            "editing agent.py / make_persist / make_finger",
            "run-time token generation",
            "donor-swap battery",
            "TM.0.17 / 0.0.005 / FAMILY / LOOKAHEAD / pixels / nursery-world",
            "earned_next=true or non-null ex0s",
            "rewriting life_wall_fixture.json",
        ],
    }
    path.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return snap


def verify_life_wall_lock(
    summary: dict[str, Any] | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    if not LIFE_WALL_LOCK.exists():
        return False, "docs/life_wall.lock missing", {}
    lock = json.loads(LIFE_WALL_LOCK.read_text(encoding="utf-8"))
    if lock.get("earned_next") is not False or lock.get("ex0s") is not None:
        return False, "earn/product drift", lock
    persist = json.loads(PERSIST_LOCK.read_text(encoding="utf-8"))
    # Historical agent_sha from the PERSIST-era freeze; later opt-in labs may extend agent.py.
    if not lock.get("agent_sha"):
        return False, "agent_sha missing", lock
    if lock.get("agent_sha") != persist.get("agent_sha"):
        return False, "agent_sha != persist.lock", lock
    if lock.get("fixture_sha") != _sha_file(FIXTURE_JSON):
        return False, "fixture_sha drift", lock
    if lock.get("life_wall_prereg_sha") != _sha_file(PREREG_LOCK):
        return False, "prereg sha drift", lock
    if not lock.get("run_tm016lifewall_sha"):
        return False, "runner sha missing", lock
    if summary is not None:
        if lock.get("last_ok_rung") != summary.get("last_ok_rung"):
            return False, "last_ok_rung drift", lock
        if lock.get("first_fail_rung") != summary.get("first_fail_rung"):
            return False, "first_fail_rung drift", lock
    return True, "life_wall.lock intact", lock


def write_results_md(summary: dict[str, Any]) -> None:
    ff = summary.get("first_fail")
    if ff is None:
        wall_line = f"Cleared through rung **{summary['last_ok_rung']}** (main + twin + forks)."
    else:
        wall_line = (
            f"Last ok rung **{summary['last_ok_rung']}**; first fail at rung "
            f"**{summary['first_fail_rung']}**: "
            f"`({ff.get('rung')}, {ff.get('lane')}, {ff.get('checkpoint')}, "
            f"{ff.get('object_index')}, {ff.get('expected')}, {ff.get('actual')})`."
        )
    lines = [
        "# TM.0.16.LIFEWALL results",
        "",
        f"**Recorded:** continuous lifetime wall → **{'PASS' if summary['ok'] else 'WALL'}**",
        "",
        "- Product: `0.0.004`",
        "- `earned_next=false`",
        "- `ex0s=null`",
        "- Organism: frozen PERSIST-on (`make_persist`); `agent.py` unchanged",
        "",
        "## Capacity",
        "",
        wall_line,
        "",
        "| Lane | ok | last_ok_rung | checkpoints |",
        "|------|----|--------------|-------------|",
        f"| main | {summary['main']['ok']} | {summary['main']['last_ok_rung']} | {summary['main']['n_checkpoints']} |",
        f"| twin | {summary['twin']['ok']} | {summary['twin']['last_ok_rung']} | {summary['twin']['n_checkpoints']} |",
        "",
        "## Bounded fact",
        "",
        "Frozen PERSIST-on coexists RELATE, ALIASFINGER, and mark-continuity in one "
        "accumulating lifetime. Alias fingerprints and continuity marks do not substitute. "
        "This is a capacity/integration wall, not a new mechanism or product stamp.",
        "",
        "## Next",
        "",
        "Identify the first grounded nursery-world channel. No product stamp.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python -m experiments.run_tm016lifewall --verify-prereg",
        "python tests/test_tm016lifewall.py",
        "```",
        "",
    ]
    RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-prereg", action="store_true")
    ap.add_argument("--write-lock", action="store_true")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    if args.verify_prereg:
        ok, why, _ = verify_prereg_lock()
        print(json.dumps({"ok": ok, "why": why}, indent=2))
        sys.exit(0 if ok else 1)

    summary = run_lifewall(seed=args.seed, write_lock=args.write_lock)
    out = {k: v for k, v in summary.items() if k not in ("main", "twin")}
    out["main_ok"] = summary["main"]["ok"]
    out["twin_ok"] = summary["twin"]["ok"]
    print(json.dumps(out, indent=2))
    sys.exit(0 if summary["ok"] else 0)  # wall failure is a valid result


if __name__ == "__main__":
    main()
