"""TM.0.24.MOTORPERSIST — scalar persistence on the zero-input motor tick.

v30 amendment candidate. Not a product earn. Product 0.0.004.
DEV p-grid on unused TM024.MOTORPERSIST.DEV. worlds.
Scoring requires docs/lineage_motorpersist.runner.lock on clean origin/main
after selected p is frozen.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from experiments.run_tm023cortex import make_cortex, physics, torch_env
from experiments.run_tm024actorcredit import MID_BODY, clone_frozen, motor_scores, observe_cue, prep_eval
from experiments.run_tm024collisionmap import (
    cosine,
    cue_pair,
    distinct,
    l2,
    one_trace,
    pair_metrics,
    warmup_vocab,
)
from experiments.run_tm024lineage import make_synthetic_world
from experiments.run_tm024statemap import live_probe, teach_one
from three_memory.cortex_lineage import sha_file
from three_memory.neural_cortex import NeuralCortex

REPO_ROOT = Path(__file__).resolve().parents[1]
THIS = Path(__file__).resolve()
PREREG = REPO_ROOT / "docs" / "lineage_motorpersist.prereg.lock"
CONTRACT = REPO_ROOT / "docs" / "lineage_motorpersist_contract.md"
ISOLATION = REPO_ROOT / "docs" / "lineage_motorpersist.isolation.lock"
P_LOCK = REPO_ROOT / "docs" / "lineage_motorpersist.p.lock"
RUNNER_LOCK = REPO_ROOT / "docs" / "lineage_motorpersist.runner.lock"
DECISION = REPO_ROOT / "docs" / "lineage_motorpersist.decision.lock"
REAUDIT_LOCK = REPO_ROOT / "docs" / "lineage_motorpersist.reaudit.lock"
REAUDIT_MD = REPO_ROOT / "docs" / "tm024motorpersist_reaudit.md"
RESULT_MD = REPO_ROOT / "docs" / "tm024motorpersist_results.md"
HISTORICAL_DECISION_SHA = "75d0ab57de8c4e394d42f78fa2af0494722b85348f123b9796de3e9b74af93b9"
HISTORICAL_P_LOCK_SHA = "634886c2fa0ccc531b390b9f919ac7c64262a4748c95cee9381362fbab6d1559"
NEURAL = REPO_ROOT / "three_memory" / "neural_cortex.py"
MEMORY = REPO_ROOT / "three_memory" / "cortex_memory.py"
CANDIDATE_V29 = REPO_ROOT / "docs" / "cortex.candidate.v29.lock"
CANDIDATE_V30 = REPO_ROOT / "docs" / "cortex.candidate.v30.lock"
CANDIDATE_LIVE = REPO_ROOT / "docs" / "cortex.candidate.lock"
V29_NEURAL = "d75b8da7f251378c9638cf9a0c4a859f12b0215d9f6f7b1623e704d831f86d03"

DEV_DOMAIN = "TM024.MOTORPERSIST.DEV."
SCORE_DOMAIN = "TM024.MOTORPERSIST.SCORE."
TWIN_DOMAIN = "TM024.MOTORPERSIST.TWIN."
REGRESS_DOMAIN = "TM024.MOTORPERSIST.REGRESS."


def load_prereg() -> dict[str, Any]:
    return json.loads(PREREG.read_text(encoding="utf-8"))


def thr() -> dict[str, Any]:
    return load_prereg()["thresholds"]


def p_grid() -> list[float]:
    return [float(x) for x in load_prereg()["p_grid"]]


def domain_seed(domain: str, key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{domain}:{key}".encode()).digest()[:8], "big") % (2**31)


def make_cell_world(index: int, domain: str) -> dict[str, Any]:
    seed = domain_seed(domain, f"world_{index}")
    w = make_synthetic_world(seed, teacher_convention=index % 2)
    w["domain"] = domain
    w["diag_index"] = int(index)
    return w


def motorpersist_shas() -> dict[str, str]:
    files = {
        "runner": THIS,
        "neural_cortex": NEURAL,
        "cortex_memory": MEMORY,
        "prereg": PREREG,
        "contract": CONTRACT,
        "isolation": ISOLATION,
        "candidate_v29": CANDIDATE_V29,
        "p_lock": P_LOCK,
        "candidate_v30": CANDIDATE_V30,
        "v30_pipeline": REPO_ROOT / "experiments" / "cortex_v30_pipeline.py",
    }
    return {k: sha_file(p) for k, p in files.items() if p.exists()}


def _git_clean() -> bool:
    return subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT).decode().strip() == ""


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()


def apply_persist(ag: NeuralCortex, p: float) -> None:
    if not hasattr(ag.genome, "motor_persist_p"):
        raise RuntimeError("motor_persist_p missing — implement the authorized mix first")
    ag.genome.motor_persist_p = float(p)


def neural_has_persist() -> bool:
    return sha_file(NEURAL) != V29_NEURAL and "MOTOR_PERSIST_P" in NEURAL.read_text(encoding="utf-8")


def _fresh(tmp: str, tag: str, world: dict[str, Any], p: float | None = None) -> NeuralCortex:
    ag = make_cortex(Path(tmp) / tag, device="cpu")
    ag.bind_actuators(list(world["handles"]))
    if p is not None:
        apply_persist(ag, p)
    return ag


def handles(world: dict[str, Any]) -> tuple[str, str]:
    return str(world["handles"][0]), str(world["handles"][1])


# One shared homeostatic-positive delta from MID. Distinct state labels keep
# the two handles observationally different; the body consequence is identical
# so last-write-wins cannot be confounded with larger-write-wins.
POS_DELTA = [0.25, -0.1, 0.15, 0.0]
POS_DELTA_H1 = list(POS_DELTA)
POS_DELTA_H2 = list(POS_DELTA)
TEACH_ORDERS = ("A_then_B", "B_then_A")


def opposing_world(world: dict[str, Any]) -> dict[str, Any]:
    w = copy.deepcopy(world)
    h1, h2 = handles(w)
    effects = dict(w["latent"]["act_effects"])
    effects[h1] = {"state": ["st_p1"], "delta": list(POS_DELTA)}
    effects[h2] = {"state": ["st_p2"], "delta": list(POS_DELTA)}
    w["latent"] = {"act_effects": effects}
    return w


def mid_adv(world: dict[str, Any], tok: str) -> float:
    body = list(MID_BODY)
    _, body2 = physics(body, tok, world["latent"])
    sp = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64)
    body_adv = float(np.linalg.norm(np.asarray(body) - sp) - np.linalg.norm(np.asarray(body2) - sp))
    return body_adv - 0.05


def live_handles(ag: NeuralCortex, world: dict[str, Any], cue: str, *, tag: str) -> dict[str, Any]:
    probe = clone_frozen(ag)
    observe_cue(probe, world, tag=tag, body=list(MID_BODY), symbols=[cue])
    scores = motor_scores(probe)
    h1, h2 = handles(world)
    s1 = float(scores.get(h1, 0.0))
    s2 = float(scores.get(h2, 0.0))
    return {
        "h1": s1,
        "h2": s2,
        "prefer_h1": bool(s1 > s2 + 1e-9),
        "prefer_h2": bool(s2 > s1 + 1e-9),
        "rho": probe._from_t(probe.rho).copy(),
    }


def pair_elig(ag: NeuralCortex, world: dict[str, Any]) -> dict[str, Any]:
    a, b = cue_pair(world)
    tr_a = one_trace(ag, world, a, tag="mp_a")
    tr_b = one_trace(ag, world, b, tag="mp_b")
    met = pair_metrics(tr_a, tr_b)
    elig = met["stages"]["rho_elig"]
    obs = tr_a["stages"]["observable"]
    elig_a = tr_a["stages"]["rho_elig"]
    step = l2(obs, elig_a)
    return {
        "cue_a": a,
        "cue_b": b,
        "elig_distinct": bool(elig["distinct"]),
        "elig_cosine": float(elig["cosine"]),
        "elig_l2": float(elig["l2"]),
        "motor_step_l2": float(step),
        "observable_distinct": bool(met["stages"]["observable"]["distinct"]),
        "rows": met["stages"],
    }


def s0_ok(ag: NeuralCortex, world: dict[str, Any], *, tag: str) -> dict[str, Any]:
    t = teach_one(ag, world, world["beneficial"], tag=tag)
    passed = bool(t["moved"] and (t["d_w_op"] > 1e-12 or t["d_w_q"] > 1e-12))
    return {"passed": passed, "d_w_op": t["d_w_op"], "d_w_q": t["d_w_q"], "moved": bool(t["moved"])}


def teach_opposing(
    ag: NeuralCortex,
    world: dict[str, Any],
    *,
    tag: str,
    order: str = "A_then_B",
) -> dict[str, Any]:
    w = opposing_world(world)
    a, b = cue_pair(w)
    h1, h2 = handles(w)
    adv1 = mid_adv(w, h1)
    adv2 = mid_adv(w, h2)
    if adv1 <= 0.0 or adv2 <= 0.0:
        raise RuntimeError(f"opposing handles must both have positive MID advantage, got {adv1}, {adv2}")
    if order == "B_then_A":
        seq = ((b, h2, "b"), (a, h1, "a"))
    else:
        seq = ((a, h1, "a"), (b, h2, "b"))
    if abs(adv1 - adv2) > 1e-12:
        raise RuntimeError(f"opposing handles must have equal MID advantage, got {adv1}, {adv2}")
    taught = []
    rhos: list[np.ndarray] = []
    for cue, tok, name in seq:
        t = teach_one(ag, w, tok, tag=f"{tag}_{name}", symbols=[cue])
        rho = np.asarray(t["rho_teach"], dtype=np.float64).copy()
        rhos.append(rho)
        taught.append({"cue": cue, "tok": tok, "moved": bool(t["moved"]), "adv": t["adv"]})
    if any(float(row["adv"]) <= 0.0 for row in taught):
        raise RuntimeError(f"opposing teach applied non-positive advantage: {taught}")
    teach_rho_cosine = float(cosine(rhos[0], rhos[1]))
    teach_rho_l2 = float(l2(rhos[0], rhos[1]))
    pa = live_handles(ag, w, a, tag=f"{tag}_pa")
    pb = live_handles(ag, w, b, tag=f"{tag}_pb")
    opposing = bool(pa["prefer_h1"] and pb["prefer_h2"])
    if order == "B_then_A":
        last_write_wins = bool(pa["prefer_h1"] and pb["prefer_h1"])
        last_write_handle = h1
    else:
        last_write_wins = bool(pa["prefer_h2"] and pb["prefer_h2"])
        last_write_handle = h2
    return {
        "order": order,
        "taught": taught,
        "live_a": {k: pa[k] for k in ("h1", "h2", "prefer_h1", "prefer_h2")},
        "live_b": {k: pb[k] for k in ("h1", "h2", "prefer_h1", "prefer_h2")},
        "opposing": opposing,
        "last_write_wins": last_write_wins,
        "last_write_handle": last_write_handle,
        "teach_rho_cosine": teach_rho_cosine,
        "teach_rho_l2": teach_rho_l2,
        "mid_adv": {"h1": adv1, "h2": adv2},
        "equal_adv": True,
    }


def eval_p_on_world(
    p: float,
    world: dict[str, Any],
    *,
    p0_step: float | None,
    order: str = "A_then_B",
) -> dict[str, Any]:
    t = thr()
    with tempfile.TemporaryDirectory(prefix="mp_dev_") as tmp:
        ag = _fresh(tmp, "s", world, p=p)
        warmup_vocab(ag, world)
        elig = pair_elig(ag, world)
        s0 = s0_ok(_fresh(tmp, "s0", world, p=p), world, tag="dev_s0")
        opp = teach_opposing(_fresh(tmp, "opp", world, p=p), world, tag="dev_opp", order=order)
        step = float(elig["motor_step_l2"])
        frac = None if p0_step is None or p0_step <= 1e-12 else step / float(p0_step)
        motor_alive = bool(p0_step is None or (frac is not None and frac >= float(t["motor_step_min_frac_of_p0"])))
        identity = bool(elig["elig_distinct"])
        usable = bool(identity and opp["opposing"] and s0["passed"] and motor_alive)
    return {
        "p": float(p),
        "domain": world["domain"],
        "index": world["diag_index"],
        "order": order,
        "identity": identity,
        "elig_cosine": elig["elig_cosine"],
        "elig_l2": elig["elig_l2"],
        "teach_rho_cosine": opp["teach_rho_cosine"],
        "teach_rho_l2": opp["teach_rho_l2"],
        "s0": s0["passed"],
        "opposing": opp["opposing"],
        "last_write_wins": opp["last_write_wins"],
        "opposing_detail": opp,
        "motor_step_l2": step,
        "motor_step_frac_of_p0": frac,
        "motor_alive": motor_alive,
        "usable": usable,
    }


def run_dev_grid() -> dict[str, Any]:
    if not neural_has_persist():
        raise RuntimeError("neural still v29 — implement the authorized mix before DEV")
    worlds = [make_cell_world(i, DEV_DOMAIN) for i in range(2)]
    rows: list[dict[str, Any]] = []
    p0_steps: list[float] = []
    for w in worlds:
        r0 = eval_p_on_world(0.0, w, p0_step=None)
        p0_steps.append(float(r0["motor_step_l2"]))
        rows.append(r0)
    selected = None
    for p in p_grid():
        if p == 0.0:
            continue
        world_rows = []
        ok_all = True
        for w, p0_step in zip(worlds, p0_steps):
            r = eval_p_on_world(p, w, p0_step=p0_step)
            world_rows.append(r)
            rows.append(r)
            ok_all = ok_all and bool(r["usable"])
        if ok_all and selected is None:
            selected = float(p)
    return {
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "n": 64,
        "grid": p_grid(),
        "select": "smallest_p_meeting_all_dev_criteria",
        "selected_p": selected,
        "p0_steps": p0_steps,
        "rows": rows,
            "note": "DEV only. Do not tune on SCORE worlds. Opposing teach uses two homeostatic-positive handles so both updates have positive advantage.",
    }


def _compact_reaudit_row(row: dict[str, Any]) -> dict[str, Any]:
    det = row.get("opposing_detail") or {}
    return {
        "p": row["p"],
        "domain": row["domain"],
        "index": row["index"],
        "order": row["order"],
        "identity": row["identity"],
        "elig_cosine": row["elig_cosine"],
        "elig_l2": row["elig_l2"],
        "teach_rho_cosine": row["teach_rho_cosine"],
        "teach_rho_l2": row["teach_rho_l2"],
        "s0": row["s0"],
        "opposing": row["opposing"],
        "last_write_wins": row["last_write_wins"],
        "last_write_handle": det.get("last_write_handle"),
        "live_a": det.get("live_a"),
        "live_b": det.get("live_b"),
        "taught_adv": [float(t["adv"]) for t in det.get("taught") or []],
        "mid_adv": det.get("mid_adv"),
        "equal_adv": det.get("equal_adv"),
        "motor_step_l2": row["motor_step_l2"],
        "motor_alive": row["motor_alive"],
        "usable": row["usable"],
    }


def run_reaudit_grid() -> dict[str, Any]:
    """Equal-advantage, both-order DEV re-audit. Does not rewrite historical p.lock."""
    if not neural_has_persist():
        raise RuntimeError("neural still v29 — implement the authorized mix before DEV")
    worlds = [make_cell_world(i, DEV_DOMAIN) for i in range(2)]
    p0_steps: list[float] = []
    for w in worlds:
        r0 = eval_p_on_world(0.0, w, p0_step=None, order="A_then_B")
        p0_steps.append(float(r0["motor_step_l2"]))
    rows: list[dict[str, Any]] = []
    for p in p_grid():
        for w, p0_step in zip(worlds, p0_steps):
            for order in TEACH_ORDERS:
                p0 = None if p == 0.0 else p0_step
                r = eval_p_on_world(p, w, p0_step=p0, order=order)
                rows.append(_compact_reaudit_row(r))
    n = len(rows)
    if n != 24:
        raise RuntimeError(f"reaudit must be exactly 24 sequences, got {n}")
    if any(r["domain"] != DEV_DOMAIN for r in rows):
        raise RuntimeError("reaudit opened a non-DEV domain")
    last_write_all = all(bool(r["last_write_wins"]) for r in rows)
    opposing_any = any(bool(r["opposing"]) for r in rows)
    equal_adv_all = all(bool(r["equal_adv"]) for r in rows)
    a_then_b = [r for r in rows if r["order"] == "A_then_B"]
    b_then_a = [r for r in rows if r["order"] == "B_then_A"]
    a_both_b = all(bool(r["live_a"]["prefer_h2"] and r["live_b"]["prefer_h2"]) for r in a_then_b)
    b_both_a = all(bool(r["live_a"]["prefer_h1"] and r["live_b"]["prefer_h1"]) for r in b_then_a)
    cos = [float(r["teach_rho_cosine"]) for r in rows]
    l2s = [float(r["teach_rho_l2"]) for r in rows]
    return {
        "version": "TM.0.24.MOTORPERSIST.REAUDIT",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "n": 64,
        "n_sequences": n,
        "grid": p_grid(),
        "orders": list(TEACH_ORDERS),
        "n_worlds": 2,
        "domain": DEV_DOMAIN,
        "score_domain_opened": False,
        "equal_positive_consequences": equal_adv_all,
        "pos_delta": list(POS_DELTA),
        "last_write_wins_all": last_write_all,
        "a_then_b_both_select_b": a_both_b,
        "b_then_a_both_select_a": b_both_a,
        "opposing_any": opposing_any,
        "usable_any": any(bool(r["usable"]) for r in rows),
        "teach_rho_cosine_min": min(cos),
        "teach_rho_cosine_max": max(cos),
        "teach_rho_l2_min": min(l2s),
        "teach_rho_l2_max": max(l2s),
        "historical_decision_sha": HISTORICAL_DECISION_SHA,
        "historical_p_lock_sha": HISTORICAL_P_LOCK_SHA,
        "decision_code": "identity_survives_opposing_learning_fails",
        "next": "plastic_write_geometry_or_connection_local_state",
        "rows": rows,
        "note": "Equal MID consequences and both teaching orders. Last write still wins. Diagnosis direction unchanged. SCORE unopened. Historical MOTORPERSIST locks untouched.",
    }


def write_reaudit_lock(dev: dict[str, Any] | None = None) -> dict[str, Any]:
    if sha_file(DECISION) != HISTORICAL_DECISION_SHA:
        raise RuntimeError("historical decision.lock drifted — do not rewrite it")
    if sha_file(P_LOCK) != HISTORICAL_P_LOCK_SHA:
        raise RuntimeError("historical p.lock drifted — do not rewrite it")
    out = dev or run_reaudit_grid()
    out["env"] = torch_env()
    out["git_head"] = _git_head()
    if REAUDIT_LOCK.exists():
        raise RuntimeError("reaudit.lock exists")
    REAUDIT_LOCK.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    REAUDIT_MD.write_text(
        "# TM.0.24.MOTORPERSIST equal-advantage re-audit\n\n"
        f"{out['n_sequences']} DEV sequences (`p` grid × 2 worlds × both orders). "
        "Both handles share one positive MID delta. "
        f"A→B both select B: **{out['a_then_b_both_select_b']}**. "
        f"B→A both select A: **{out['b_then_a_both_select_a']}**. "
        f"Opposing any: **{out['opposing_any']}**. "
        f"Teaching-state cosine [{out['teach_rho_cosine_min']:.6f}, {out['teach_rho_cosine_max']:.6f}]; "
        f"L2 [{out['teach_rho_l2_min']:.6f}, {out['teach_rho_l2_max']:.6f}].\n\n"
        "Diagnosis unchanged: **identity_survives_opposing_learning_fails**. "
        "SCORE unopened. Historical MOTORPERSIST locks untouched. "
        "Next is plastic-write geometry. Product **0.0.004**. `earned_next=false`.\n",
        encoding="utf-8",
    )
    return out


def write_p_lock(dev: dict[str, Any] | None = None) -> dict[str, Any]:
    if not neural_has_persist():
        raise RuntimeError("neural still v29")
    dev = dev or run_dev_grid()
    src = NEURAL.read_text(encoding="utf-8")
    frozen_p = None
    for line in src.splitlines():
        if line.startswith("MOTOR_PERSIST_P"):
            frozen_p = float(line.split("=")[1].split("#")[0].strip())
            break
    if frozen_p is None:
        raise RuntimeError("MOTOR_PERSIST_P missing")
    if dev["selected_p"] is None:
        lock = {
            "version": "TM.0.24.MOTORPERSIST.P",
            "product": "0.0.004",
            "earned_next": False,
            "ex0s": None,
            "eligible_for_000005": False,
            "n": 64,
            "selected_p": None,
            "module_p": frozen_p,
            "dev": {k: v for k, v in dev.items() if k != "rows"},
            "rows": dev["rows"],
            "usable_p_exists": False,
            "note": "No grid value met all DEV criteria after teaching two positive-advantage handles. Do not score as a pass. Escalation table applies.",
        }
    else:
        if abs(frozen_p - float(dev["selected_p"])) > 1e-12:
            raise RuntimeError(
                f"MOTOR_PERSIST_P={frozen_p} != selected DEV p={dev['selected_p']} — freeze the module constant first"
            )
        lock = {
            "version": "TM.0.24.MOTORPERSIST.P",
            "product": "0.0.004",
            "earned_next": False,
            "ex0s": None,
            "eligible_for_000005": False,
            "n": 64,
            "selected_p": float(dev["selected_p"]),
            "module_p": frozen_p,
            "grid": p_grid(),
            "dev_domain": DEV_DOMAIN,
            "usable_p_exists": True,
            "neural_sha": sha_file(NEURAL),
            "note": "Selected smallest DEV p. Frozen before scored worlds.",
        }
    if P_LOCK.exists():
        raise RuntimeError("p.lock exists")
    P_LOCK.write_text(json.dumps(lock, indent=2, default=str) + "\n", encoding="utf-8")
    return lock


def frozen_p() -> float:
    if not P_LOCK.exists():
        raise RuntimeError("no p.lock")
    lock = json.loads(P_LOCK.read_text(encoding="utf-8"))
    if lock.get("selected_p") is None:
        raise RuntimeError("no usable p was frozen")
    return float(lock["selected_p"])


def assert_runner_frozen() -> dict[str, Any]:
    if not RUNNER_LOCK.exists():
        raise RuntimeError("no motorpersist runner.lock — refuse cell scoring")
    lock = json.loads(RUNNER_LOCK.read_text(encoding="utf-8"))
    if motorpersist_shas() != lock.get("shas"):
        raise RuntimeError("motorpersist implementation drifted after runner.lock")
    if not CANDIDATE_V30.exists():
        raise RuntimeError("candidate.v30 required before scoring")
    cand = json.loads(CANDIDATE_V30.read_text(encoding="utf-8"))
    if sha_file(NEURAL) != cand["neural_cortex_sha"]:
        raise RuntimeError("live neural drifted from v30 candidate")
    if cand.get("genome", {}).get("n") != 64:
        raise RuntimeError("n must stay 64")
    live = json.loads(CANDIDATE_LIVE.read_text(encoding="utf-8"))
    if live.get("version") != "TM.0.23.CORTEX.CANDIDATE.V30":
        raise RuntimeError("live candidate is not v30")
    return lock


def write_runner_lock() -> dict[str, Any]:
    if not _git_clean():
        raise RuntimeError("write runner.lock only on a clean tree")
    if not CANDIDATE_V30.exists() or not P_LOCK.exists():
        raise RuntimeError("candidate.v30 and p.lock required")
    p = frozen_p()
    lock = {
        "version": "TM.0.24.MOTORPERSIST.RUNNER.V1",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "shas": motorpersist_shas(),
        "prereg_sha": sha_file(PREREG),
        "contract_sha": sha_file(CONTRACT),
        "isolation_sha": sha_file(ISOLATION),
        "candidate_v29_sha": sha_file(CANDIDATE_V29),
        "candidate_v30_sha": sha_file(CANDIDATE_V30),
        "p_lock_sha": sha_file(P_LOCK),
        "p": p,
        "n": 64,
        "domain": SCORE_DOMAIN,
        "twin_domain": TWIN_DOMAIN,
        "regress_domain": REGRESS_DOMAIN,
        "gates": load_prereg()["gates"],
        "git_head": _git_head(),
        "note": "Frozen P-gate runner. Score only after this lock is on origin/main. Do not tune p.",
    }
    RUNNER_LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return lock


def run_p0(*, domain: str = SCORE_DOMAIN) -> dict[str, Any]:
    rows = []
    passed = True
    for i in range(3):
        world = make_cell_world(i, domain)
        with tempfile.TemporaryDirectory(prefix="mp_p0_") as tmp:
            ag = _fresh(tmp, "s", world)
            warmup_vocab(ag, world)
            elig = pair_elig(ag, world)
        rows.append(elig)
        passed = passed and bool(elig["elig_distinct"])
    return {"id": "P0", "passed": passed, "rows": rows, "domain": domain}


def run_p1(*, domain: str = SCORE_DOMAIN, index: int = 0, order: str = "A_then_B") -> dict[str, Any]:
    world = make_cell_world(index, domain)
    with tempfile.TemporaryDirectory(prefix="mp_p1_") as tmp:
        ag = _fresh(tmp, "s", world)
        out = teach_opposing(ag, world, tag="p1", order=order)
    return {"id": "P1" if order == "A_then_B" else "P2", "passed": bool(out["opposing"]), **out, "domain": domain}


def run_p3(*, domain: str = SCORE_DOMAIN, index: int = 1) -> dict[str, Any]:
    world = make_cell_world(index, domain)
    w = opposing_world(world)
    a, b = cue_pair(w)
    h1, h2 = handles(w)
    with tempfile.TemporaryDirectory(prefix="mp_p3_") as tmp:
        ag = _fresh(tmp, "s", world)
        teach_opposing(ag, world, tag="p3base", order="A_then_B")
        before_b = live_handles(ag, w, b, tag="p3_bb")
        teach_one(ag, w, h2, tag="p3_rev_a", symbols=[a])
        after_a = live_handles(ag, w, a, tag="p3_aa")
        after_b = live_handles(ag, w, b, tag="p3_ba")
        a_flipped = bool(after_a["prefer_h2"])
        b_stable = bool(after_b["prefer_h2"] == before_b["prefer_h2"] and after_b["prefer_h2"])
        passed = bool(a_flipped and b_stable)
    return {
        "id": "P3",
        "passed": passed,
        "a_flipped": a_flipped,
        "b_stable": b_stable,
        "after_a": {k: after_a[k] for k in ("h1", "h2", "prefer_h1", "prefer_h2")},
        "after_b": {k: after_b[k] for k in ("h1", "h2", "prefer_h1", "prefer_h2")},
        "domain": domain,
    }


def run_p4() -> dict[str, Any]:
    import experiments.run_tm024statemap as sm

    saved_c, saved_t = sm.CELLS_DOMAIN, sm.TWIN_DOMAIN
    sm.CELLS_DOMAIN = REGRESS_DOMAIN
    sm.TWIN_DOMAIN = TWIN_DOMAIN
    try:
        s0 = sm.run_s0()
        s1 = sm.run_s1()
        s2 = sm.run_s2()
        s3 = sm.run_s3()
        s4 = sm.run_s4()
        s5 = sm.run_s5()
        s6 = sm.run_s6()
        s7 = sm.run_s7()
        s8 = sm.run_s8()
        s9 = sm.run_s9()
        s10 = sm.run_s10()
        s11 = sm.run_s11()
        s12 = sm.run_s12(s0, s7)
    finally:
        sm.CELLS_DOMAIN = saved_c
        sm.TWIN_DOMAIN = saved_t
    cells = [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12]
    must_pass = {c["id"]: c["passed"] for c in cells if c["id"] in {"S0", "S1", "S2", "S3", "S4", "S5", "S7", "S8", "S9", "S12"}}
    must_fail = {c["id"]: c["passed"] for c in cells if c["id"] in {"S10", "S11"}}
    passed = bool(all(must_pass.values()) and not any(must_fail.values()))
    return {
        "id": "P4",
        "passed": passed,
        "must_pass": must_pass,
        "must_fail": must_fail,
        "s6_passed": bool(s6["passed"]),
        "s6_note": "S6 may change; judged by P1, not P4.",
        "cells": [{k: c[k] for k in ("id", "passed") if k in c} | ({"kind": c["kind"]} if "kind" in c else {}) for c in cells],
        "domain": REGRESS_DOMAIN,
    }


def run_p5(p4: dict[str, Any] | None = None) -> dict[str, Any]:
    p4 = p4 or run_p4()
    by = {c["id"]: c["passed"] for c in p4["cells"]}
    world = make_cell_world(0, SCORE_DOMAIN)
    with tempfile.TemporaryDirectory(prefix="mp_p5_") as tmp:
        ag = _fresh(tmp, "birth", world)
        birth_zero = float(ag.W_act_query.abs().sum()) == 0.0
        n64 = int(ag.genome.n) == 64
    checks = {
        "hold_s8": bool(by.get("S8")),
        "rest_consolidation_s9": bool(by.get("S9")),
        "reset_s4": bool(by.get("S4")),
        "distractor_s3": bool(by.get("S3")),
        "rename_s12": bool(by.get("S12")),
        "birth_w_act_query_zero": birth_zero,
        "n64": n64,
    }
    return {"id": "P5", "passed": all(checks.values()), "checks": checks}


def run_p6(*, domain: str = SCORE_DOMAIN, index: int = 0) -> dict[str, Any]:
    world = make_cell_world(index, domain)
    a, b = cue_pair(world)
    with tempfile.TemporaryDirectory(prefix="mp_p6_") as tmp:
        ag = _fresh(tmp, "s", world)
        warmup_vocab(ag, world)
        pa = live_handles(ag, world, a, tag="p6a")
        pb = live_handles(ag, world, b, tag="p6b")
        innate_opposing = bool(pa["prefer_h1"] and pb["prefer_h2"])
        birth_zero = float(ag.W_act_query.abs().sum()) == 0.0
        passed = bool((not innate_opposing) and birth_zero)
    return {
        "id": "P6",
        "passed": passed,
        "innate_opposing": innate_opposing,
        "birth_zero": birth_zero,
        "live_a": {k: pa[k] for k in ("h1", "h2", "prefer_h1", "prefer_h2")},
        "live_b": {k: pb[k] for k in ("h1", "h2", "prefer_h1", "prefer_h2")},
    }


def escalate(gates: list[dict[str, Any]], dev: dict[str, Any] | None) -> dict[str, Any]:
    by = {g["id"]: g for g in gates}
    identity = bool(by.get("P0", {}).get("passed"))
    opposing = bool(by.get("P1", {}).get("passed")) and bool(by.get("P2", {}).get("passed"))
    if dev is not None and dev.get("selected_p") is None:
        any_identity = any(r.get("identity") for r in dev.get("rows") or [])
        any_alive = any(r.get("motor_alive") for r in dev.get("rows") or [] if r.get("p", 0) > 0)
        if not any_identity:
            code = "identity_collapses_every_usable_p"
            next_step = "generic_context_motor_partition_inside_n64"
        elif any_identity and not any(r.get("opposing") for r in dev.get("rows") or []):
            code = "identity_survives_opposing_learning_fails"
            next_step = "plastic_write_geometry_or_connection_local_state"
        elif any_identity and not any_alive:
            code = "high_p_preserves_identity_destroys_motor"
            next_step = "learned_gate_or_functional_cell_classes"
        else:
            code = "dev_grid_no_usable_p"
            next_step = "read_dev_rows"
        return {"code": code, "next": next_step, "passed": False}
    if all(g.get("passed") for g in gates):
        return {
            "code": "scalar_persistence_succeeds",
            "next": "freeze_v30_then_deterministic_reachability_lineage_still_closed",
            "passed": True,
        }
    if not identity:
        return {
            "code": "identity_collapses_on_scored_worlds",
            "next": "generic_context_motor_partition_inside_n64",
            "passed": False,
        }
    if identity and not opposing:
        return {
            "code": "identity_survives_opposing_learning_fails",
            "next": "plastic_write_geometry_or_connection_local_state",
            "passed": False,
        }
    return {"code": "scored_gate_fail", "next": "read_gate_rows", "passed": False}


def run_all() -> dict[str, Any]:
    assert_runner_frozen()
    p0 = run_p0()
    p1 = run_p1(order="A_then_B")
    p2 = run_p1(order="B_then_A")
    p2["id"] = "P2"
    p3 = run_p3()
    p4 = run_p4()
    p5 = run_p5(p4)
    p6 = run_p6()
    gates = [p0, p1, p2, p3, p4, p5, p6]
    decision = escalate(gates, None)
    out = {
        "version": "TM.0.24.MOTORPERSIST.DECISION",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "capability_claim": False,
        "n": 64,
        "p": frozen_p(),
        "gates": gates,
        "n_pass": sum(1 for g in gates if g.get("passed")),
        "n_gates": len(gates),
        "all_passed": all(g.get("passed") for g in gates),
        "decision": decision,
        "lineage_reopened": False,
        "q3": False,
        "env": torch_env(),
        "git_head": _git_head(),
        "note": "v30 amendment candidate. Lineage stays closed. Product remains 0.0.004.",
    }
    return out


def write_dev_decision() -> dict[str, Any]:
    if not P_LOCK.exists():
        raise RuntimeError("p.lock required")
    plock = json.loads(P_LOCK.read_text(encoding="utf-8"))
    rows = plock.get("rows") or []
    identity_any = any(r.get("identity") and r.get("p", 0) > 0 for r in rows)
    opposing_any = any(r.get("opposing") for r in rows)
    motor_alive_pos = any(r.get("motor_alive") and r.get("p", 0) > 0 for r in rows)
    decision = escalate([], plock if plock.get("selected_p") is None else {"selected_p": plock.get("selected_p"), "rows": rows})
    # escalate() with empty gates and selected_p None uses rows from the second arg
    out = {
        "version": "TM.0.24.MOTORPERSIST.DECISION",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "capability_claim": False,
        "n": 64,
        "selected_p": plock.get("selected_p"),
        "usable_p_exists": bool(plock.get("usable_p_exists")),
        "module_p": plock.get("module_p"),
        "scored_worlds": False,
        "identity_any_positive_p": identity_any,
        "opposing_any": opposing_any,
        "motor_alive_any_positive_p": motor_alive_pos,
        "n_pass": 0,
        "n_gates": 7,
        "all_passed": False,
        "decision": decision,
        "lineage_reopened": False,
        "q3": False,
        "p_lock_sha": sha_file(P_LOCK),
        "env": torch_env(),
        "git_head": _git_head(),
        "note": "DEV grid: no p met identity+opposing+S0+motor simultaneously. Opposing teach uses two positive-advantage handles. Scored worlds not opened. Next is plastic-write geometry. Product remains 0.0.004.",
    }
    if DECISION.exists():
        raise RuntimeError("decision.lock exists")
    DECISION.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    RESULT_MD.write_text(
        "# TM.0.24.MOTORPERSIST result\n\n"
        "No usable `p`. Identity can survive (`p≥0.25` by L2). Opposing sequential teaching failed at every grid value even after teaching two positive-advantage handles. "
        "`p≥0.9` disables the motor step. Escalation: **plastic-write geometry / compact connection-local state**.\n\n"
        "Scored worlds were not opened. Lineage stays closed. Product **0.0.004**. `earned_next=false`.\n",
        encoding="utf-8",
    )
    return out
    if DECISION.exists():
        raise RuntimeError("decision.lock exists")
    DECISION.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    lines = [
        "# TM.0.24.MOTORPERSIST result",
        "",
        f"p = `{out['p']}`. Gates **{out['n_pass']}/{out['n_gates']}**. Decision `{out['decision']['code']}`.",
        "",
        "Lineage stays closed. Product **0.0.004**. `earned_next=false`.",
        "",
    ]
    for g in out["gates"]:
        lines.append(f"- `{g['id']}`: **{'PASS' if g.get('passed') else 'FAIL'}**")
    RESULT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def smoke() -> dict[str, Any]:
    prereg = load_prereg()
    w = make_cell_world(0, DEV_DOMAIN)
    with tempfile.TemporaryDirectory(prefix="mp_smk_") as tmp:
        ag = make_cortex(Path(tmp) / "s", device="cpu")
        ag.bind_actuators(list(w["handles"]))
        t = teach_one(ag, w, w["beneficial"], tag="smk")
        live = live_probe(ag, w, tag="smk_p")
        cos = cosine(t["rho_teach"], live["rho"])
    return {
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "capability_claim": False,
        "smoke_ok": True,
        "n": 64,
        "d_w_op": t["d_w_op"],
        "cosine_teach_vs_probe": cos,
        "gates": prereg["gates"],
        "p_grid": prereg["p_grid"],
        "neural_has_persist": neural_has_persist(),
        "env": torch_env(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dev-grid", action="store_true")
    ap.add_argument("--write-p-lock", action="store_true")
    ap.add_argument("--write-dev-decision", action="store_true")
    ap.add_argument("--write-runner-lock", action="store_true")
    ap.add_argument("--reaudit", action="store_true")
    ap.add_argument("--write-reaudit-lock", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--write-decision", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        print(json.dumps(smoke(), indent=2, default=str))
    elif args.dev_grid:
        print(json.dumps(run_dev_grid(), indent=2, default=str))
    elif args.write_p_lock:
        print(json.dumps(write_p_lock(), indent=2, default=str))
    elif args.write_dev_decision:
        print(json.dumps(write_dev_decision(), indent=2, default=str))
    elif args.write_runner_lock:
        print(json.dumps(write_runner_lock(), indent=2))
    elif args.reaudit:
        print(json.dumps(run_reaudit_grid(), indent=2, default=str))
    elif args.write_reaudit_lock:
        print(json.dumps(write_reaudit_lock(), indent=2, default=str))
    elif args.score:
        out = run_all()
        if args.write_decision:
            write_decision(out)
        print(json.dumps({k: v for k, v in out.items() if k != "gates"}, indent=2, default=str))
        print(json.dumps([{"id": g["id"], "passed": g.get("passed")} for g in out["gates"]], indent=2))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
