"""TM.0.24.PHASEMAP — runner-only causal-phase address map.

Not a lineage version. Not a capability earn. No neural edit. Product 0.0.004.
Write-geometry branch closed. Frozen D1 from DISCRIMMAP R2 is the instrument.
DEV on unused TM024.PHASEMAP.DEV. after this freeze is on origin/main.
SCORE reserved and unopened.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from experiments.run_tm023cortex import physics, torch_env
from experiments.run_tm024actorcredit import MID_BODY, observe_cue, prep_eval
from experiments.run_tm024collisionmap import parse_stages
from experiments.run_tm024discrimmap_r2 import (
    ACCEPTED_STATUS,
    hard_margin_linear,
    min_geometric_margin,
    perturb_sign_stable,
    require_accepted,
)
from experiments.run_tm024eligmap import (
    _fresh,
    capacity_world,
    mapping_pairs,
    record_observe,
    unit_or_zero,
)
from experiments.run_tm024motorpersist import TEACH_ORDERS
from three_memory.cortex_lineage import sha_file
from three_memory.neural_cortex import NeuralCortex

REPO_ROOT = Path(__file__).resolve().parents[1]
THIS = Path(__file__).resolve()
PREREG = REPO_ROOT / "docs" / "lineage_phasemap.prereg.lock"
CONTRACT = REPO_ROOT / "docs" / "lineage_phasemap_contract.md"
ISOLATION = REPO_ROOT / "docs" / "lineage_phasemap.isolation.lock"
RUNNER_LOCK = REPO_ROOT / "docs" / "lineage_phasemap.runner.lock"
DEV_LOCK = REPO_ROOT / "docs" / "lineage_phasemap.dev.lock"
DECISION = REPO_ROOT / "docs" / "lineage_phasemap.decision.lock"
RESULT_MD = REPO_ROOT / "docs" / "tm024phasemap_results.md"
R2_PREREG = REPO_ROOT / "docs" / "lineage_discrimmap.r2.prereg.lock"
R2_DEC = REPO_ROOT / "docs" / "lineage_discrimmap.r2.decision.lock"
R2_ADD = REPO_ROOT / "docs" / "lineage_discrimmap.r2.decision.addendum.lock"
R2_RUNNER = REPO_ROOT / "experiments" / "run_tm024discrimmap_r2.py"
NEURAL = REPO_ROOT / "three_memory" / "neural_cortex.py"
MEMORY = REPO_ROOT / "three_memory" / "cortex_memory.py"
CANDIDATE_V30 = REPO_ROOT / "docs" / "cortex.candidate.v30.lock"
CANDIDATE_V31 = REPO_ROOT / "docs" / "cortex.candidate.v31.lock"

DEV_DOMAIN = "TM024.PHASEMAP.DEV."
TWIN_DOMAIN = "TM024.PHASEMAP.TWIN."
SCORE_DOMAIN = "TM024.PHASEMAP.SCORE."
SCORE_MARKERS = ("TM024.PHASEMAP.SCORE.",)
EPS = 1e-12
PHASES = ("P0", "P1", "P2", "P3", "P4", "P5")
EXPECTED_N_RANK = 6 * 3 * 2 * 2
EXPECTED_N_TWIN = 6 * 2
EXPECTED_N_CELLS = EXPECTED_N_RANK + EXPECTED_N_TWIN


def load_prereg() -> dict[str, Any]:
    return json.loads(PREREG.read_text(encoding="utf-8"))


def d1_spec() -> dict[str, Any]:
    spec = json.loads(R2_PREREG.read_text(encoding="utf-8"))["arms"]["D1"]
    if spec.get("soft_margin") or spec.get("soft_margin_C") is not None:
        raise RuntimeError("D1 oracle acquired a soft-margin degree of freedom")
    return spec


def phasemap_shas() -> dict[str, str]:
    files = {
        "runner": THIS,
        "neural_cortex": NEURAL,
        "cortex_memory": MEMORY,
        "prereg": PREREG,
        "contract": CONTRACT,
        "isolation": ISOLATION,
        "candidate_v30": CANDIDATE_V30,
        "r2_prereg": R2_PREREG,
        "r2_decision": R2_DEC,
        "r2_addendum": R2_ADD,
        "r2_runner": R2_RUNNER,
    }
    return {k: sha_file(p) for k, p in files.items() if p.exists()}


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()


def cell_id(kind: str, phase: str, n_cues: int, order: str, world: int) -> str:
    return f"{kind}|{phase}|c{n_cues}|{order}|w{world}"


def assert_runner_frozen() -> dict[str, Any]:
    if not RUNNER_LOCK.exists():
        raise RuntimeError("no phasemap runner.lock — refuse DEV lock")
    lock = json.loads(RUNNER_LOCK.read_text(encoding="utf-8"))
    if phasemap_shas() != lock.get("shas"):
        raise RuntimeError("preregistration or runner hashes mismatch after runner.lock")
    if lock.get("n") != 64:
        raise RuntimeError("n must stay 64")
    if sha_file(PREREG) != lock["shas"]["prereg"]:
        raise RuntimeError("preregistration hash mismatch")
    return lock


def refuse_rerun() -> None:
    if DEV_LOCK.exists():
        raise RuntimeError("same frozen DEV execution requested again")


def refuse_score_markers(payload: str) -> None:
    for mark in SCORE_MARKERS:
        if mark in payload:
            raise RuntimeError("SCORE identifier appeared in DEV payload")


def phase_bundle(ag: NeuralCortex) -> dict[str, np.ndarray]:
    st = parse_stages(ag)
    pre = np.asarray(st["observable"], dtype=np.float64).copy()
    return {
        "P0": unit_or_zero(st["cue"]),
        "P1": unit_or_zero(st["event_end"]),
        "P2": unit_or_zero(st["observable"]),
        "P3": unit_or_zero(pre),
        "P4": unit_or_zero(st["motor_last"]),
        "P5": unit_or_zero(st["rho_elig"]),
        "n_motor": int(st["_n_motor"][0]),
        "n_sensory": int(st["_n_sensory"][0]),
    }


def teach_select_phases(
    ag: NeuralCortex, world: dict[str, Any], cue: str, handle: str, *, tag: str
) -> tuple[dict[str, np.ndarray], float]:
    record_observe(ag, world, kind="select", tag=f"{tag}_sel", cue=cue, handle=handle)
    stages = phase_bundle(ag)
    ag.clamp_action("ACT", handle)
    _, body2 = physics(list(MID_BODY), handle, world["latent"])
    cred_out = observe_cue(ag, world, tag=f"{tag}_obs", body=list(body2), symbols=[cue])
    prep_eval(ag)
    adv = float((cred_out.get("metrics") or {}).get("adv") or 0.0)
    return stages, adv


def probe_phases(
    ag: NeuralCortex, world: dict[str, Any], cue: str, *, tag: str
) -> dict[str, np.ndarray]:
    record_observe(ag, world, kind="probe", tag=tag, cue=cue)
    return phase_bundle(ag)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= EPS or nb <= EPS:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def pairwise_stats(vecs: list[np.ndarray], handles: list[str]) -> dict[str, float]:
    n = len(vecs)
    between: list[float] = []
    between_diff: list[float] = []
    bcos: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = l2(vecs[i], vecs[j])
            c = cosine(vecs[i], vecs[j])
            between.append(d)
            bcos.append(c)
            if handles[i] != handles[j]:
                between_diff.append(d)
    return {
        "between_l2_min": float(min(between)) if between else 0.0,
        "between_l2_mean": float(sum(between) / len(between)) if between else 0.0,
        "between_l2_min_diff_handle": float(min(between_diff)) if between_diff else 0.0,
        "between_cos_max": float(max(bcos)) if bcos else 1.0,
        "between_cos_min": float(min(bcos)) if bcos else 1.0,
    }


def within_stats(repeats: list[list[np.ndarray]]) -> dict[str, float]:
    dists: list[float] = []
    coss: list[float] = []
    for reps in repeats:
        for i in range(len(reps)):
            for j in range(i + 1, len(reps)):
                dists.append(l2(reps[i], reps[j]))
                coss.append(cosine(reps[i], reps[j]))
    return {
        "within_l2_mean": float(sum(dists) / len(dists)) if dists else 0.0,
        "within_l2_max": float(max(dists)) if dists else 0.0,
        "within_cos_min": float(min(coss)) if coss else 1.0,
    }


def contraction_stats(
    prev: list[np.ndarray], cur: list[np.ndarray]
) -> dict[str, float | None]:
    ratios: list[float] = []
    for i in range(len(prev)):
        for j in range(i + 1, len(prev)):
            den = l2(prev[i], prev[j])
            if den <= EPS:
                continue
            ratios.append(l2(cur[i], cur[j]) / den)
    if not ratios:
        return {"contraction_min": None, "contraction_mean": None, "n_pairs": 0}
    return {
        "contraction_min": float(min(ratios)),
        "contraction_mean": float(sum(ratios) / len(ratios)),
        "n_pairs": len(ratios),
    }


def collect_block(
    world: dict[str, Any],
    pairs: list[tuple[str, str]],
    *,
    order: str,
    tag: str,
    n_repeats: int,
) -> dict[str, Any]:
    seq = list(reversed(pairs)) if order == "B_then_A" else list(pairs)
    handles = list(world["handles"])
    with tempfile.TemporaryDirectory(prefix="pm_") as tmp:
        ag = _fresh(tmp, "s", world)
        teach: list[dict[str, Any]] = []
        for i, (cue, handle) in enumerate(seq):
            stages, adv = teach_select_phases(ag, world, cue, handle, tag=f"{tag}_t{i}")
            if abs(float(adv)) <= EPS:
                raise RuntimeError("zero advantage on teach select")
            teach.append({"cue": cue, "handle": handle, "adv": float(adv), "stages": stages})
        teach_by_cue = {t["cue"]: t for t in teach}
        probes: list[dict[str, Any]] = []
        repeats: dict[str, list[dict[str, np.ndarray]]] = {c: [] for c, _h in pairs}
        for i, (cue, handle) in enumerate(pairs):
            st = probe_phases(ag, world, cue, tag=f"{tag}_p{i}")
            probes.append({"cue": cue, "handle": handle, "stages": st})
            repeats[cue].append(st)
            for r in range(1, int(n_repeats)):
                repeats[cue].append(probe_phases(ag, world, cue, tag=f"{tag}_r{i}_{r}"))
    ordered_teach = [teach_by_cue[c] for c, _h in pairs]
    return {
        "teach": ordered_teach,
        "probes": probes,
        "repeats": repeats,
        "handles": handles,
        "pairs": pairs,
        "n_motor": [int(t["stages"]["n_motor"]) for t in ordered_teach],
    }


def eval_phase(
    block: dict[str, Any],
    *,
    phase: str,
    prev: str | None,
    world: dict[str, Any],
    tag: str,
) -> dict[str, Any]:
    p = load_prereg()
    spec = d1_spec()
    gmin = float(p["margin"]["geometric_margin_min"])
    handles = list(block["handles"])
    h0 = handles[0]
    teach_x = [t["stages"][phase] for t in block["teach"]]
    teach_y = np.asarray([1.0 if t["handle"] == h0 else -1.0 for t in block["teach"]])
    teach_h = [str(t["handle"]) for t in block["teach"]]
    probe_x = [q["stages"][phase] for q in block["probes"]]
    probe_y = np.asarray([1.0 if q["handle"] == h0 else -1.0 for q in block["probes"]])
    X_tr = np.stack(teach_x)
    X_te = np.stack(probe_x)
    if X_tr is X_te:
        raise RuntimeError("probe rows passed to fit")
    fit = hard_margin_linear(X_tr, teach_y, spec)
    require_accepted(fit["status"] if fit["status"] in ACCEPTED_STATUS else "error")
    w, b, status = fit["w"], fit["b"], fit["status"]
    train_g = min_geometric_margin(w, b, X_tr, teach_y)
    probe_g = min_geometric_margin(w, b, X_te, probe_y)
    train_rank = bool(len(teach_y) and np.all((X_tr @ w + b) * teach_y > 0.0))
    ranking_ok = bool(len(probe_y) and np.all((X_te @ w + b) * probe_y > 0.0))
    train_clean = bool(status == "optimal" and train_rank and train_g >= gmin)
    stab = {"stable": False, "n_ok": 0, "n": 20}
    if status == "optimal":
        stab = perturb_sign_stable(w, b, X_te, probe_y, domain=world["domain"], key=f"{tag}_{phase}")
    passed = bool(
        status == "optimal"
        and train_rank
        and ranking_ok
        and train_g >= gmin
        and probe_g >= gmin
        and stab["stable"]
    )
    sep = pairwise_stats(teach_x, teach_h)
    rep_vecs = [[st[phase] for st in block["repeats"][c]] for c, _h in block["pairs"]]
    within = within_stats(rep_vecs)
    ctr = {"contraction_min": None, "contraction_mean": None, "n_pairs": 0}
    if prev is not None:
        ctr = contraction_stats([t["stages"][prev] for t in block["teach"]], teach_x)
    sep_thr = p["separation"]
    same_cue_stable = bool(
        within["within_l2_max"] <= float(sep_thr["within_l2_stable_max"])
        and within["within_cos_min"] >= float(sep_thr["within_cos_stable_min"])
    )
    between_distinct = bool(
        sep["between_l2_min"] > float(sep_thr["l2_distinct_min"])
        or sep["between_cos_max"] < float(sep_thr["cos_distinct_max"])
    )
    return {
        "phase": phase,
        "passed": passed,
        "solver_status": status,
        "n_sv": int(fit["n_sv"]),
        "ranking_ok": ranking_ok,
        "train_ranking_ok": train_rank,
        "train_clean": train_clean,
        "train_geometric_margin": float(train_g),
        "probe_geometric_margin": float(probe_g),
        "perturb_stable": bool(stab["stable"]),
        "perturb_n_ok": int(stab.get("n_ok") or 0),
        "n_train": int(len(teach_y)),
        "n_probe": int(len(probe_y)),
        "n_repeats": int(p["n_repeats"]),
        "w_norm": float(np.linalg.norm(w)),
        "same_cue_stable": same_cue_stable,
        "between_distinct": between_distinct,
        **within,
        **sep,
        **ctr,
        "n_motor_mean": float(sum(block["n_motor"]) / max(len(block["n_motor"]), 1)),
        "p2_equals_p3": bool(
            phase in ("P2", "P3")
            and np.allclose(block["teach"][0]["stages"]["P2"], block["teach"][0]["stages"]["P3"])
        ),
    }


def smoke() -> dict[str, Any]:
    p = load_prereg()
    spec = d1_spec()
    assert spec["no_automatic_soft_margin"] is True
    world = capacity_world(0, "TM024.PHASEMAP.SMOKE.", n_cues=2, n_handles=2)
    pairs = mapping_pairs(world, flip=False)
    block = collect_block(world, pairs, order="A_then_B", tag="pmsmk", n_repeats=int(p["n_repeats"]))
    p0 = eval_phase(block, phase="P0", prev=None, world=world, tag="smk")
    p4 = eval_phase(block, phase="P4", prev="P3", world=world, tag="smk")
    p5 = eval_phase(block, phase="P5", prev="P4", world=world, tag="smk")
    return {
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "smoke_ok": True,
        "n": 64,
        "p2_equals_p3": p0["p2_equals_p3"] or eval_phase(block, phase="P2", prev="P1", world=world, tag="smk")["p2_equals_p3"],
        "p0_status": p0["solver_status"],
        "p0_train_g": p0["train_geometric_margin"],
        "p4_contraction_mean": p4["contraction_mean"],
        "p5_status": p5["solver_status"],
        "expected_n_cells": EXPECTED_N_CELLS,
        "neural_edit": False,
        "v31_exists": CANDIDATE_V31.exists(),
        "write_geometry_branch_closed": True,
        "eligibility_budget_installed": False,
        "env": torch_env(),
    }


def _decision(cells: list[dict[str, Any]], p: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    gmin = float(p["margin"]["geometric_margin_min"])

    def rank8(phase: str) -> list[dict[str, Any]]:
        return [
            c
            for c in cells
            if c["phase"] == phase and c["kind"] == "rank" and c["n_cues"] == 8
        ]

    def robust(phase: str) -> bool:
        rank = rank8(phase)
        twin = [c for c in cells if c["phase"] == phase and c["kind"] == "twin"]
        return bool(rank) and bool(twin) and all(c["passed"] for c in rank + twin)

    def train_clean(phase: str) -> bool:
        rank = rank8(phase)
        return bool(rank) and all(bool(c["train_clean"]) for c in rank)

    def probe_transfer_fail(phase: str) -> bool:
        rank = rank8(phase)
        return bool(rank) and any(
            (not bool(c["ranking_ok"])) or float(c["probe_geometric_margin"]) < gmin
            for c in rank
        )

    flags = {
        ph: {
            "robust": robust(ph),
            "train_clean": train_clean(ph),
            "probe_transfer_fail": probe_transfer_fail(ph),
        }
        for ph in PHASES
    }
    r = [flags[ph]["robust"] for ph in PHASES]
    ladder = p["decision_ladder"]
    first_fail = next((ph for ph in PHASES if not flags[ph]["robust"]), None)
    if all(r):
        return ladder[0]["id"], ladder[0]["then"], {"flags": flags, "first_fail": None}
    if (
        first_fail is not None
        and flags[first_fail]["train_clean"]
        and flags[first_fail]["probe_transfer_fail"]
    ):
        return ladder[1]["id"], ladder[1]["then"], {"flags": flags, "first_fail": first_fail}
    if r[0] and r[1] and r[2] and r[3] and r[4] and not r[5]:
        return ladder[2]["id"], ladder[2]["then"], {"flags": flags, "first_fail": "P5"}
    if r[0] and r[1] and r[2] and r[3] and not r[4]:
        return ladder[3]["id"], ladder[3]["then"], {"flags": flags, "first_fail": "P4"}
    if r[0] and not all(r):
        return ladder[4]["id"], ladder[4]["then"], {"flags": flags, "first_fail": first_fail}
    return ladder[5]["id"], ladder[5]["then"], {"flags": flags, "first_fail": first_fail or "P0"}


def run_dev() -> dict[str, Any]:
    if CANDIDATE_V31.exists():
        raise RuntimeError("v31 candidate must not exist")
    refuse_rerun()
    lock = assert_runner_frozen()
    p = load_prereg()
    if sha_file(PREREG) != lock["shas"]["prereg"]:
        raise RuntimeError("preregistration hash mismatch")
    n_repeats = int(p["n_repeats"])
    cells: list[dict[str, Any]] = []
    for spec in p["capacity"]:
        n_cues = int(spec["n_cues"])
        for wi in range(2):
            world = capacity_world(wi, DEV_DOMAIN, n_cues=n_cues, n_handles=2)
            if SCORE_DOMAIN in world["domain"] or "SCORE." in world["domain"]:
                raise RuntimeError("SCORE identifier appeared in DEV payload")
            pairs = mapping_pairs(world, flip=False)
            for order in TEACH_ORDERS:
                block = collect_block(
                    world, pairs, order=order, tag=f"pm_{wi}_{n_cues}_{order}", n_repeats=n_repeats
                )
                prev = None
                for ph in PHASES:
                    out = eval_phase(
                        block, phase=ph, prev=prev, world=world, tag=f"rank{wi}_{n_cues}_{order}"
                    )
                    out.update(
                        {
                            "id": cell_id("rank", ph, n_cues, order, wi),
                            "kind": "rank",
                            "order": order,
                            "n_cues": n_cues,
                            "n_handles": 2,
                            "world": wi,
                            "required": True,
                            "domain": world["domain"],
                        }
                    )
                    cells.append(out)
                    prev = ph
    twin_cells: list[dict[str, Any]] = []
    for order in TEACH_ORDERS:
        world = capacity_world(1, TWIN_DOMAIN, n_cues=2, n_handles=2)
        world["purpose"] = "rename_twin"
        if SCORE_DOMAIN in world["domain"] or "SCORE." in world["domain"]:
            raise RuntimeError("SCORE identifier appeared in DEV payload")
        pairs = mapping_pairs(world, flip=False)
        block = collect_block(world, pairs, order=order, tag=f"pm_twin_{order}", n_repeats=n_repeats)
        prev = None
        for ph in PHASES:
            out = eval_phase(block, phase=ph, prev=prev, world=world, tag=f"twin_{order}")
            out.update(
                {
                    "id": cell_id("twin", ph, 2, order, 1),
                    "kind": "twin",
                    "order": order,
                    "n_cues": 2,
                    "n_handles": 2,
                    "world": 1,
                    "required": True,
                    "domain": world["domain"],
                }
            )
            twin_cells.append(out)
            prev = ph
    all_cells = cells + twin_cells
    ids = [c["id"] for c in all_cells]
    if len(ids) != EXPECTED_N_CELLS or len(set(ids)) != EXPECTED_N_CELLS:
        raise RuntimeError(f"missing or duplicated cell {len(ids)} unique {len(set(ids))}")
    for c in all_cells:
        if int(c["n_train"]) != int(c["n_cues"]) or int(c["n_probe"]) != int(c["n_cues"]):
            raise RuntimeError(f"empty teach/probe {c['id']}")
        if c["domain"] not in (DEV_DOMAIN, TWIN_DOMAIN):
            raise RuntimeError(f"unexpected domain {c['domain']}")
        require_accepted(str(c["solver_status"]))
    code, then, extra = _decision(all_cells, p)
    flags = extra["flags"]
    out = {
        "version": "TM.0.24.PHASEMAP.DEV",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "n": 64,
        "domain": DEV_DOMAIN,
        "twin_domain": TWIN_DOMAIN,
        "score_domain_opened": False,
        "neural_edit": False,
        "implementation_authorized": False,
        "write_geometry_branch_closed": True,
        "phase_robust_8cue": {ph: flags[ph]["robust"] for ph in PHASES},
        "phase_train_clean_8cue": {ph: flags[ph]["train_clean"] for ph in PHASES},
        "phase_probe_transfer_fail_8cue": {ph: flags[ph]["probe_transfer_fail"] for ph in PHASES},
        "first_fail": extra["first_fail"],
        "decision_code": code,
        "decision_then": then,
        "n_cells": len(all_cells),
        "n_rank": len(cells),
        "n_twin": len(twin_cells),
        "cells": all_cells,
        "env": torch_env(),
        "git_head": _git_head(),
        "shas": phasemap_shas(),
        "note": "PHASEMAP DEV only. Write-geometry closed. No neural edit. Product remains 0.0.004.",
    }
    refuse_score_markers(json.dumps(out, default=str))
    return out


def write_runner_lock() -> dict[str, Any]:
    if RUNNER_LOCK.exists():
        raise RuntimeError("phasemap runner.lock already exists")
    if CANDIDATE_V31.exists():
        raise RuntimeError("v31 candidate must not exist")
    prereg = load_prereg()
    lock = {
        "version": "TM.0.24.PHASEMAP.RUNNER.V1",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "neural_edit": False,
        "implementation_authorized": False,
        "write_geometry_branch_closed": True,
        "shas": phasemap_shas(),
        "n": 64,
        "domain": DEV_DOMAIN,
        "twin_domain": TWIN_DOMAIN,
        "score_domain": SCORE_DOMAIN,
        "score_reserved_unopened": True,
        "phases": list(PHASES),
        "phase_keys": prereg["phase_keys"],
        "d1_oracle": prereg["d1_oracle"],
        "geometric_margin_min": prereg["margin"]["geometric_margin_min"],
        "n_repeats": prereg["n_repeats"],
        "contraction": prereg["contraction"]["formula"],
        "expected_n_rank": EXPECTED_N_RANK,
        "expected_n_twin": EXPECTED_N_TWIN,
        "expected_n_cells": EXPECTED_N_CELLS,
        "fail_closed": prereg["fail_closed"],
        "decision_ladder": [r["then"] for r in prereg["decision_ladder"]],
        "git_head": _git_head(),
        "note": "Frozen PHASEMAP runner. DEV lock only after this file is on origin/main. No neural edit.",
    }
    RUNNER_LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return lock


def write_dev_lock(out: dict[str, Any]) -> dict[str, Any]:
    assert_runner_frozen()
    refuse_rerun()
    refuse_score_markers(json.dumps(out, default=str))
    DEV_LOCK.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    return out


def write_decision(dev: dict[str, Any]) -> dict[str, Any]:
    if DECISION.exists():
        raise RuntimeError("phasemap decision lock already exists")
    out = {
        "version": "TM.0.24.PHASEMAP.DECISION",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "capability_claim": False,
        "n": 64,
        "scored_worlds": False,
        "neural_edit": False,
        "implementation_authorized": False,
        "candidate_v31": False,
        "lineage_reopened": False,
        "q3": False,
        "eligibility_budget_installed": False,
        "declared_budget_remains_closed": 1536,
        "write_geometry_branch_closed": True,
        "decision": {
            "code": dev["decision_code"],
            "then": dev["decision_then"],
            "first_fail": dev.get("first_fail"),
            "phase_robust_8cue": dev.get("phase_robust_8cue"),
            "phase_train_clean_8cue": dev.get("phase_train_clean_8cue"),
            "phase_probe_transfer_fail_8cue": dev.get("phase_probe_transfer_fail_8cue"),
        },
        "dev_lock_sha": sha_file(DEV_LOCK) if DEV_LOCK.exists() else None,
        "env": dev.get("env"),
        "git_head": _git_head(),
        "note": (
            "Runner-only causal phase map. Write-geometry closed. "
            "No v31. Lineage stays closed. Product remains 0.0.004."
        ),
    }
    DECISION.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    RESULT_MD.write_text(
        "# TM.0.24.PHASEMAP DEV\n\n"
        f"Decision: **{out['decision']['code']}**. "
        f"First fail: **{out['decision']['first_fail']}**.\n\n"
        f"Eight-cue robust: `{out['decision']['phase_robust_8cue']}`.\n\n"
        "Write-geometry closed. SCORE unopened. No neural candidate. "
        "1536 eligibility budget stays closed. Product **0.0.004**. `earned_next=false`.\n",
        encoding="utf-8",
    )
    return out


def refuse_score() -> None:
    raise RuntimeError("SCORE opens only after PHASEMAP identifies a robust causal source state")


def refuse_dev_lock() -> None:
    if not RUNNER_LOCK.exists():
        raise RuntimeError("PHASEMAP DEV lock requires runner.lock on origin/main after this freeze")
    assert_runner_frozen()
    refuse_rerun()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--verify-prereg", action="store_true")
    ap.add_argument("--write-runner-lock", action="store_true")
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--write-dev-lock", action="store_true")
    ap.add_argument("--write-decision", action="store_true")
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        print(json.dumps(smoke(), indent=2, default=str))
    elif args.verify_prereg:
        p = load_prereg()
        assert p["n"] == 64
        assert p["neural_edit"] is False
        assert p["write_geometry_branch_closed"] is True
        assert p["phases"] == list(PHASES)
        assert p["d1_oracle"]["no_new_solver"] is True
        assert p["margin"]["geometric_margin_min"] == 0.01
        assert p["declared_budget_remains_closed"] == 1536
        print(json.dumps({"ok": True, "product": p["product"], "expected_n_cells": EXPECTED_N_CELLS}, indent=2))
    elif args.write_runner_lock:
        print(json.dumps(write_runner_lock(), indent=2, default=str))
    elif args.dev:
        out = run_dev()
        print(json.dumps({k: v for k, v in out.items() if k != "cells"}, indent=2, default=str))
    elif args.write_dev_lock:
        refuse_dev_lock()
        out = run_dev()
        write_dev_lock(out)
        print(json.dumps({k: v for k, v in out.items() if k != "cells"}, indent=2, default=str))
    elif args.write_decision:
        if not DEV_LOCK.exists():
            raise RuntimeError("write DEV lock first")
        dev = json.loads(DEV_LOCK.read_text(encoding="utf-8"))
        print(json.dumps(write_decision(dev), indent=2, default=str))
    elif args.score:
        refuse_score()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
