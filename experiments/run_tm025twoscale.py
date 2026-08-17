"""TM.0.25.TWOSCALE — v32 fast episodic P1 plus slow consolidation battery.

Not a product earn. Product 0.0.004. SCORE reserved.
DEV on unused TM025.TWOSCALE.DEV. / TWIN.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from experiments.run_tm023cortex import make_cortex, torch_env
from experiments.run_tm024actorcredit import MID_BODY, clone_frozen, observe_cue
from experiments.run_tm024convergencemap import unique_winner
from experiments.run_tm024statemap import teach_one
from experiments.run_tm024writegeom import (
    NEG_DELTA,
    capacity_world,
    mapping_pairs,
    ranking_margin,
    set_handle_delta,
)
from three_memory.cortex_lineage import sha_file
from three_memory.neural_cortex import (
    ACT_MARGIN_FLOOR,
    ACT_SCORE_QUERY,
    EPISODE_MATCH_L2,
    EPISODE_REPLAY_EPOCHS,
    EPISODE_SLOTS,
    NeuralCortex,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
THIS = Path(__file__).resolve()
PREREG = REPO_ROOT / "docs" / "lineage_twoscale.prereg.lock"
CONTRACT = REPO_ROOT / "docs" / "lineage_twoscale_contract.md"
ISOLATION = REPO_ROOT / "docs" / "lineage_twoscale.isolation.lock"
V32_PREREG = REPO_ROOT / "docs" / "cortex_v32.prereg.lock"
V32_ISO = REPO_ROOT / "docs" / "cortex_v32.isolation.lock"
V32_AMEND = REPO_ROOT / "docs" / "cortex_v32_architecture_amendment.lock"
DEV_LOCK = REPO_ROOT / "docs" / "lineage_twoscale.dev.lock"
DECISION = REPO_ROOT / "docs" / "lineage_twoscale.decision.lock"
RESULT_MD = REPO_ROOT / "docs" / "tm025twoscale_results.md"
NEURAL = REPO_ROOT / "three_memory" / "neural_cortex.py"
MEMORY = REPO_ROOT / "three_memory" / "cortex_memory.py"

DEV_DOMAIN = "TM025.TWOSCALE.DEV."
TWIN_DOMAIN = "TM025.TWOSCALE.TWIN."
SCORE_DOMAIN = "TM025.TWOSCALE.SCORE."
EXPECTED_N_CELLS = 36


def load_prereg() -> dict[str, Any]:
    return json.loads(PREREG.read_text(encoding="utf-8"))


def domain_seed(domain: str, key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{domain}:{key}".encode()).digest()[:8], "big") % (2**31)


def twoscale_shas() -> dict[str, str]:
    files = {
        "runner": THIS,
        "neural_cortex": NEURAL,
        "cortex_memory": MEMORY,
        "prereg": PREREG,
        "contract": CONTRACT,
        "isolation": ISOLATION,
        "v32_prereg": V32_PREREG,
        "v32_isolation": V32_ISO,
        "v32_amendment": V32_AMEND,
        "affinemap_r2_decision": REPO_ROOT / "docs" / "lineage_affinemap.r2.decision.lock",
        "affinemap_r2_addendum": REPO_ROOT / "docs" / "lineage_affinemap.r2.decision.addendum.lock",
        "candidate_v30": REPO_ROOT / "docs" / "cortex.candidate.v30.lock",
    }
    return {k: sha_file(p) for k, p in files.items() if p.exists()}


def refuse_score() -> None:
    raise RuntimeError("TM.0.25.TWOSCALE SCORE is reserved and must not be opened")


def refuse_dev_lock() -> None:
    if DEV_LOCK.exists():
        raise RuntimeError("TM.0.25.TWOSCALE DEV lock exists; same frozen DEV execution refused again")
    if not V32_PREREG.exists():
        raise RuntimeError("TWOSCALE DEV lock must wait for cortex_v32.prereg.lock")


def _fresh(tmp: str, tag: str, world: dict[str, Any]) -> NeuralCortex:
    ag = make_cortex(Path(tmp) / tag, device="cpu")
    ag.bind_actuators(list(world["handles"]))
    if str(ag.genome.act_score_mode) != ACT_SCORE_QUERY:
        raise RuntimeError("v32 default scoring must remain query")
    return ag


def p1_probe(ag: NeuralCortex, world: dict[str, Any], cue: str, *, tag: str) -> dict[str, Any]:
    probe = clone_frozen(ag)
    observe_cue(probe, world, tag=tag, body=list(MID_BODY), symbols=[cue])
    p1 = probe._last_p1
    if p1 is None:
        p1 = probe._from_t(probe.rho)
    scores = probe.actuator_scores(p1)
    winner = unique_winner(scores)
    gap = ranking_margin(scores, winner) if winner else 0.0
    gamma = float(probe._act_geometric_margin(p1, winner)) if winner else 0.0
    return {
        "scores": {k: float(v) for k, v in scores.items()},
        "winner": winner,
        "pairwise_score_gap": float(gap),
        "normalized_geometric_margin": float(gamma),
        "p1": np.asarray(p1, dtype=np.float64).copy(),
        "n_episodes": len(probe._episodes),
    }


def perturb_p1(ag: NeuralCortex, p1: np.ndarray, want: str, *, domain: str, key: str) -> dict[str, Any]:
    m = load_prereg()["margin"]
    sigma = float(m["rho_perturb_sigma"])
    n = int(m["perturb_n"])
    need = int(m["perturb_stable_min"])
    rng = np.random.default_rng(domain_seed(domain, key))
    r0 = np.asarray(p1, dtype=np.float64).reshape(-1)
    nrm = float(np.linalg.norm(r0)) + 1e-12
    r_hat = r0 / nrm
    n_ok = 0
    for _i in range(n):
        unit = r_hat + rng.normal(0.0, sigma, size=r_hat.shape)
        pn = float(np.linalg.norm(unit)) + 1e-12
        unit = unit / pn
        scores = ag.actuator_scores(unit)
        if unique_winner(scores) == want:
            n_ok += 1
    return {"n_ok": n_ok, "n": n, "stable": n_ok >= need}


def teach_pairs(ag: NeuralCortex, world: dict[str, Any], pairs: list[tuple[str, str]], *, tag: str) -> list[dict[str, Any]]:
    taught = []
    for i, (cue, handle) in enumerate(pairs):
        t = teach_one(ag, world, handle, tag=f"{tag}_{i}", symbols=[cue])
        taught.append({"cue": cue, "handle": handle, "adv": float(t["adv"]), "n_episodes": len(ag._episodes)})
    return taught


def probe_map(ag: NeuralCortex, world: dict[str, Any], pairs: list[tuple[str, str]], *, tag: str, domain: str) -> dict[str, Any]:
    probes = []
    ranking_ok = True
    gammas = []
    gaps = []
    pert_ok = True
    for i, (cue, handle) in enumerate(pairs):
        live = p1_probe(ag, world, cue, tag=f"{tag}_p{i}")
        rank = bool(live["winner"] == handle)
        ranking_ok = ranking_ok and rank
        g = float(ag._act_geometric_margin(live["p1"], handle))
        gammas.append(g)
        gaps.append(float(live["pairwise_score_gap"]))
        stab = perturb_p1(ag, live["p1"], handle, domain=domain, key=f"{tag}_{cue}")
        pert_ok = pert_ok and bool(stab["stable"])
        probes.append(
            {
                "cue": cue,
                "want": handle,
                "winner": live["winner"],
                "ranking_ok": rank,
                "normalized_geometric_margin": g,
                "pairwise_score_gap": float(live["pairwise_score_gap"]),
                "perturbation_ok": bool(stab["stable"]),
            }
        )
    min_g = min(gammas) if gammas else 0.0
    geometric_ok = bool(min_g >= float(load_prereg()["margin"]["geometric_min"]))
    return {
        "probes": probes,
        "ranking_ok": ranking_ok,
        "perturbation_ok": pert_ok,
        "geometric_ok": geometric_ok,
        "min_normalized_geometric_margin": float(min_g),
        "min_pairwise_score_gap": float(min(gaps) if gaps else 0.0),
        "n_episodes": len(ag._episodes),
    }


def expected_cell_ids() -> list[str]:
    p = load_prereg()
    ids: list[str] = []
    orders = list(p["orders"])
    n_worlds = int(p["n_worlds"])
    for spec in p["capacity"]:
        n = int(spec["n_cues"])
        for order in orders:
            for wi in range(n_worlds):
                ids.append(f"acquire|c{n}|{order}|w{wi}")
                ids.append(f"stable|c{n}|{order}|w{wi}")
    for order in orders:
        for wi in range(n_worlds):
            ids.append(f"twin|c2|{order}|w{wi}")
            ids.append(f"eco|{order}|w{wi}")
            ids.append(f"spec|{order}|w{wi}")
    return ids


def eval_acquire_stable(
    *,
    kind: str,
    world: dict[str, Any],
    pairs: list[tuple[str, str]],
    order: str,
    tag: str,
    rest: bool,
) -> dict[str, Any]:
    seq = list(reversed(pairs)) if order == "B_then_A" else list(pairs)
    p = load_prereg()
    with tempfile.TemporaryDirectory(prefix="ts_cap_") as tmp:
        ag = _fresh(tmp, "s", world)
        taught = teach_pairs(ag, world, seq, tag=tag)
        rest_out: dict[str, Any] | None = None
        if rest:
            rest_out = ag.rest_epoch(int(p["n_rest_ticks"]))
        probed = probe_map(ag, world, pairs, tag=tag, domain=str(world["domain"]))
    ranking_ok = bool(probed["ranking_ok"])
    if kind == "acquire":
        passed = ranking_ok
    else:
        passed = bool(ranking_ok and probed["geometric_ok"] and probed["perturbation_ok"])
    return {
        "kind": kind,
        "order": order,
        "n_cues": int(world["capacity"]["n_cues"]),
        "taught": taught,
        "rest": rest_out,
        "passed": passed,
        "pass_statistic": "normalized_geometric_margin",
        **probed,
    }


def eval_eco(world: dict[str, Any], *, order: str, tag: str) -> dict[str, Any]:
    cue = world["cue_handle"][0]["cue"]
    h1, h2 = world["handles"][0], world["handles"][1]
    if order == "B_then_A":
        h1, h2 = h2, h1
    p = load_prereg()
    with tempfile.TemporaryDirectory(prefix="ts_eco_") as tmp:
        ag = _fresh(tmp, "s", world)
        t1 = teach_one(ag, world, h1, tag=f"{tag}_p", symbols=[cue])
        wneg = set_handle_delta(world, h1, NEG_DELTA)
        t2 = teach_one(ag, wneg, h1, tag=f"{tag}_n", symbols=[cue])
        t3 = teach_one(ag, world, h2, tag=f"{tag}_r", symbols=[cue])
        n_replaced = int(ag._episode_n_replaced)
        rest_out = ag.rest_epoch(int(p["n_rest_ticks"]))
        live = p1_probe(ag, world, cue, tag=f"{tag}_q")
        g = float(ag._act_geometric_margin(live["p1"], h2))
        stab = perturb_p1(ag, live["p1"], h2, domain=str(world["domain"]), key=f"{tag}_eco")
        ranking_ok = bool(live["winner"] == h2)
        geometric_ok = bool(g >= float(p["margin"]["geometric_min"]))
        passed = bool(
            t1["adv"] > 0.0
            and t2["adv"] < 0.0
            and t3["adv"] > 0.0
            and ranking_ok
            and geometric_ok
            and stab["stable"]
        )
    return {
        "kind": "eco",
        "order": order,
        "n_cues": 2,
        "passed": passed,
        "ranking_ok": ranking_ok,
        "geometric_ok": geometric_ok,
        "perturbation_ok": bool(stab["stable"]),
        "normalized_geometric_margin": g,
        "min_normalized_geometric_margin": g,
        "pairwise_score_gap": float(live["pairwise_score_gap"]),
        "winner": live["winner"],
        "want": h2,
        "adv": [float(t1["adv"]), float(t2["adv"]), float(t3["adv"])],
        "n_replaced": n_replaced,
        "rest": rest_out,
        "n_episodes": live["n_episodes"],
        "pass_statistic": "normalized_geometric_margin",
    }


def eval_spec(world: dict[str, Any], *, order: str, tag: str) -> dict[str, Any]:
    pairs = mapping_pairs(world, flip=False)
    if len(pairs) < 2:
        raise RuntimeError("spec needs two cues")
    (c_a, h_a), (c_b, h_b) = pairs[0], pairs[1]
    if order == "B_then_A":
        (c_a, h_a), (c_b, h_b) = (c_b, h_b), (c_a, h_a)
    p = load_prereg()
    with tempfile.TemporaryDirectory(prefix="ts_spec_") as tmp:
        ag = _fresh(tmp, "s", world)
        teach_one(ag, world, h_a, tag=f"{tag}_a", symbols=[c_a])
        teach_one(ag, world, h_b, tag=f"{tag}_b", symbols=[c_b])
        wneg = set_handle_delta(world, h_a, NEG_DELTA)
        teach_one(ag, wneg, h_a, tag=f"{tag}_n", symbols=[c_a])
        teach_one(ag, world, h_b, tag=f"{tag}_ar", symbols=[c_a])
        rest_out = ag.rest_epoch(int(p["n_rest_ticks"]))
        pa = p1_probe(ag, world, c_a, tag=f"{tag}_qa")
        pb = p1_probe(ag, world, c_b, tag=f"{tag}_qb")
        ga = float(ag._act_geometric_margin(pa["p1"], h_b))
        gb = float(ag._act_geometric_margin(pb["p1"], h_b))
        sa = perturb_p1(ag, pa["p1"], h_b, domain=str(world["domain"]), key=f"{tag}_a")
        sb = perturb_p1(ag, pb["p1"], h_b, domain=str(world["domain"]), key=f"{tag}_b")
        ranking_ok = bool(pa["winner"] == h_b and pb["winner"] == h_b)
        gmin = min(ga, gb)
        geometric_ok = bool(gmin >= float(p["margin"]["geometric_min"]))
        pert_ok = bool(sa["stable"] and sb["stable"])
        passed = bool(ranking_ok and geometric_ok and pert_ok)
    return {
        "kind": "spec",
        "order": order,
        "n_cues": 2,
        "passed": passed,
        "ranking_ok": ranking_ok,
        "geometric_ok": geometric_ok,
        "perturbation_ok": pert_ok,
        "min_normalized_geometric_margin": float(gmin),
        "normalized_geometric_margin": float(gmin),
        "a_winner": pa["winner"],
        "b_winner": pb["winner"],
        "want_a": h_b,
        "want_b": h_b,
        "rest": rest_out,
        "n_episodes": pa["n_episodes"],
        "pass_statistic": "normalized_geometric_margin",
    }


def _decision(cells: list[dict[str, Any]], p: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    def kind_ok(kind: str, n: int | None = None) -> bool:
        rows = [c for c in cells if c["kind"] == kind and (n is None or int(c.get("n_cues") or 0) == n)]
        return bool(rows) and all(bool(c["passed"]) for c in rows)

    flags = {
        "acquire_2": kind_ok("acquire", 2),
        "acquire_4": kind_ok("acquire", 4),
        "acquire_8": kind_ok("acquire", 8),
        "stable_2": kind_ok("stable", 2),
        "stable_4": kind_ok("stable", 4),
        "stable_8": kind_ok("stable", 8),
        "twin": kind_ok("twin"),
        "eco": kind_ok("eco"),
        "spec": kind_ok("spec"),
    }
    if not flags["acquire_2"] or not flags["acquire_4"] or not flags["acquire_8"]:
        return "architectural_wall_acquire", "architectural_wall_acquire", flags
    if not flags["stable_2"] or not flags["stable_4"] or not flags["stable_8"] or not flags["twin"]:
        return "architectural_wall_stability", "architectural_wall_stability", flags
    if not flags["eco"]:
        return "architectural_wall_reversal", "architectural_wall_reversal", flags
    if not flags["spec"]:
        return "architectural_wall_specificity", "architectural_wall_specificity", flags
    return "two_timescale_battery_pass", "reopen_lineage_readiness", flags


def run_dev() -> dict[str, Any]:
    refuse_dev_lock()
    p = load_prereg()
    cells: list[dict[str, Any]] = []
    for spec in p["capacity"]:
        n_cues = int(spec["n_cues"])
        n_handles = int(spec["n_handles"])
        for wi in range(int(p["n_worlds"])):
            world = capacity_world(wi, DEV_DOMAIN, n_cues=n_cues, n_handles=n_handles)
            pairs = mapping_pairs(world, flip=False)
            for order in list(p["orders"]):
                acq = eval_acquire_stable(
                    kind="acquire",
                    world=world,
                    pairs=pairs,
                    order=order,
                    tag=f"acq_c{n_cues}_{order}_w{wi}",
                    rest=False,
                )
                acq["id"] = f"acquire|c{n_cues}|{order}|w{wi}"
                acq["world"] = wi
                acq["domain"] = DEV_DOMAIN
                cells.append(acq)
                st = eval_acquire_stable(
                    kind="stable",
                    world=world,
                    pairs=pairs,
                    order=order,
                    tag=f"st_c{n_cues}_{order}_w{wi}",
                    rest=True,
                )
                st["id"] = f"stable|c{n_cues}|{order}|w{wi}"
                st["world"] = wi
                st["domain"] = DEV_DOMAIN
                cells.append(st)
    for wi in range(int(p["n_worlds"])):
        twin_w = capacity_world(wi, TWIN_DOMAIN, n_cues=2, n_handles=2)
        pairs = mapping_pairs(twin_w, flip=False)
        for order in list(p["orders"]):
            tw = eval_acquire_stable(
                kind="acquire",
                world=twin_w,
                pairs=pairs,
                order=order,
                tag=f"twin_{order}_w{wi}",
                rest=False,
            )
            tw["kind"] = "twin"
            tw["id"] = f"twin|c2|{order}|w{wi}"
            tw["world"] = wi
            tw["domain"] = TWIN_DOMAIN
            cells.append(tw)
            eco_w = capacity_world(wi, DEV_DOMAIN, n_cues=2, n_handles=2)
            eco = eval_eco(eco_w, order=order, tag=f"eco_{order}_w{wi}")
            eco["id"] = f"eco|{order}|w{wi}"
            eco["world"] = wi
            eco["domain"] = DEV_DOMAIN
            cells.append(eco)
            spec = eval_spec(eco_w, order=order, tag=f"spec_{order}_w{wi}")
            spec["id"] = f"spec|{order}|w{wi}"
            spec["world"] = wi
            spec["domain"] = DEV_DOMAIN
            cells.append(spec)
    ids = [str(c["id"]) for c in cells]
    expect = expected_cell_ids()
    if sorted(ids) != sorted(expect):
        raise RuntimeError(f"cell id mismatch {len(ids)} vs {len(expect)}")
    if SCORE_DOMAIN in json.dumps(cells):
        raise RuntimeError("SCORE domain leaked into DEV")
    code, then, flags = _decision(cells, p)
    env = torch_env()
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    return {
        "version": "TM.0.25.TWOSCALE.DEV",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "n": 64,
        "n_cells": len(cells),
        "domain": DEV_DOMAIN,
        "twin_domain": TWIN_DOMAIN,
        "score_domain_opened": False,
        "neural_edit": True,
        "act_score_mode": ACT_SCORE_QUERY,
        "episode_slots": EPISODE_SLOTS,
        "match_l2": EPISODE_MATCH_L2,
        "replay_epochs": EPISODE_REPLAY_EPOCHS,
        "pass_statistic": "normalized_geometric_margin",
        "geometric_min": ACT_MARGIN_FLOOR,
        "decision_code": code,
        "decision_then": then,
        "phase_flags": flags,
        "cells": cells,
        "shas": twoscale_shas(),
        "env": env,
        "git_head": git_head,
        "w1_resurrected": False,
        "a3_bias": False,
        "note": "v32 two-timescale organism battery. SCORE unopened. Product remains 0.0.004.",
    }


def write_dev_lock(out: dict[str, Any]) -> None:
    refuse_dev_lock()
    DEV_LOCK.write_text(json.dumps(out, indent=2, default=_json_default) + "\n", encoding="utf-8")


def write_decision(out: dict[str, Any]) -> None:
    dec = {
        "version": "TM.0.25.TWOSCALE.DECISION",
        "product": "0.0.004",
        "earned_next": False,
        "ex0s": None,
        "eligible_for_000005": False,
        "n": 64,
        "neural_edit": True,
        "pass_statistic": "normalized_geometric_margin",
        "decision": {
            "code": out["decision_code"],
            "then": out["decision_then"],
            "phase_flags": out["phase_flags"],
        },
        "dev_lock_sha": hashlib.sha256(DEV_LOCK.read_bytes()).hexdigest() if DEV_LOCK.exists() else None,
        "git_head": out.get("git_head"),
        "lineage_reopened": False,
        "note": "v32 two-timescale battery. SCORE unopened. Product remains 0.0.004.",
    }
    DECISION.write_text(json.dumps(dec, indent=2) + "\n", encoding="utf-8")


def write_results(out: dict[str, Any]) -> None:
    if RESULT_MD.exists():
        return
    flags = out["phase_flags"]
    lines = [
        "# TM.0.25.TWOSCALE DEV",
        "",
        f"Decision: **{out['decision_code']}**.",
        "",
        "v32 fast episodic P1 plus slow consolidation. Unused `TM025.TWOSCALE.DEV.` / `TWIN.`. "
        "SCORE unopened. Product **0.0.004**. `earned_next=false`. Lineage stays closed.",
        "",
        f"Phase flags: `{flags}`.",
        "",
        "Same frozen DEV execution refused.",
        "",
    ]
    RESULT_MD.write_text("\n".join(lines), encoding="utf-8")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(type(obj))


def smoke() -> dict[str, Any]:
    p = load_prereg()
    assert p["n"] == 64
    assert p["episode_slots"] == EPISODE_SLOTS
    assert len(expected_cell_ids()) == EXPECTED_N_CELLS
    world = capacity_world(0, DEV_DOMAIN, n_cues=2, n_handles=2)
    with tempfile.TemporaryDirectory(prefix="ts_smk_") as tmp:
        ag = _fresh(tmp, "s", world)
        assert len(ag._episodes) == 0
        assert float(ag.W_act_query.abs().max().item()) == 0.0
        cue, handle = world["cue_handle"][0]["cue"], world["cue_handle"][0]["handle"]
        t = teach_one(ag, world, handle, tag="smk", symbols=[cue])
        assert abs(float(t["adv"])) > 0.0
        assert len(ag._episodes) >= 1
        assert ag._last_p1 is not None
        rest = ag.rest_epoch(int(p["n_rest_ticks"]))
        live = p1_probe(ag, world, cue, tag="smk_p")
        snap = ag.checkpoint()
        twin = NeuralCortex(None, genome=ag.genome, device="cpu")
        twin.load_checkpoint(snap)
        twin.bind_actuators(list(world["handles"]))
        assert len(twin._episodes) == len(ag._episodes)
        assert twin._last_p1 is not None
    return {
        "smoke_ok": True,
        "expected_id_count": EXPECTED_N_CELLS,
        "n_episodes": int(live["n_episodes"]),
        "winner": live["winner"],
        "want": handle,
        "ranking_ok": bool(live["winner"] == handle),
        "adv": float(t["adv"]),
        "n_replay": int(rest.get("n_replay") or 0),
        "n_strengthen": int(rest.get("n_strengthen") or 0),
        "neural_edit": True,
        "v31_exists": (REPO_ROOT / "docs" / "cortex.candidate.v31.lock").exists(),
        "v32_candidate_exists": (REPO_ROOT / "docs" / "cortex.candidate.v32.lock").exists(),
        "tau": float(ag.genome.tau),
        "n": int(ag.genome.n),
        "act_score_mode": str(ag.genome.act_score_mode),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--write-dev-lock", action="store_true")
    ap.add_argument("--run-dev", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        print(json.dumps(smoke(), indent=2, default=_json_default))
        return
    if args.write_dev_lock and not args.run_dev:
        refuse_dev_lock()
        raise RuntimeError("refuse_dev_lock: --write-dev-lock without --run-dev does not write")
    if args.run_dev:
        out = run_dev()
        write_dev_lock(out)
        write_decision(out)
        write_results(out)
        print(json.dumps({"decision": out["decision_code"], "n_cells": out["n_cells"], "flags": out["phase_flags"]}, indent=2))
        return
    refuse_score()


if __name__ == "__main__":
    main()
