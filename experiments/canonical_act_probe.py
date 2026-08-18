"""Canonical ACT probe: the same path as organism motor scoring.

Motor ACT uses NeuralCortex.actuator_decision_scores(live_p1).
Behavioral experiments must call this module. Raw actuator_scores(live_p1)
is diagnostic_raw_live_scores only — never a behavioral gate.

No neural, SOCP, recall, or threshold changes.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path
from typing import Any

import numpy as np

from experiments.run_tm024actorcredit import MID_BODY, clone_frozen, observe_cue
from experiments.run_tm024convergencemap import unique_winner
from experiments.run_tm024writegeom import ranking_margin
from experiments.run_tm027gatedrehearsal import domain_seed
from experiments.run_tm031halfspace import arr_sha, probe_geometric_margin
from three_memory.neural_cortex import NeuralCortex

REPO_ROOT = Path(__file__).resolve().parents[1]
NEURAL = REPO_ROOT / "three_memory" / "neural_cortex.py"
BEHAVIORAL_RUNNERS = (
    REPO_ROOT / "experiments" / "canonical_act_probe.py",
    REPO_ROOT / "experiments" / "run_tm040causal_r2.py",
)


def decision_scores(
    ag: NeuralCortex,
    live_p1: np.ndarray,
    *,
    key_rho: np.ndarray | None = None,
) -> tuple[dict[str, float], np.ndarray, dict[str, Any]]:
    """Pure wrapper: identical to organism motor ACT scoring."""
    scores, addr, meta = ag.actuator_decision_scores(live_p1, key_rho=key_rho)
    return scores, np.asarray(addr, dtype=np.float64), dict(meta)


def diagnostic_raw_live_scores(ag: NeuralCortex, live_p1: np.ndarray) -> dict[str, Any]:
    """Explicitly labeled diagnostic. Never a behavioral gate."""
    scores = ag.actuator_scores(live_p1)
    winner = unique_winner(scores)
    return {
        "diagnostic": True,
        "not_a_behavioral_gate": True,
        "scores": {k: float(v) for k, v in scores.items()},
        "winner": winner,
        "live_p1_hash": arr_sha(live_p1),
    }


def canonical_cue_probe(
    ag: NeuralCortex,
    world: dict[str, Any],
    cue: str,
    *,
    tag: str,
    want: str | None = None,
) -> dict[str, Any]:
    probe = clone_frozen(ag)
    observe_cue(probe, world, tag=tag, body=list(MID_BODY), symbols=[cue])
    live_p1 = probe._last_p1
    if live_p1 is None:
        live_p1 = probe._from_t(probe.rho)
    live_p1 = np.asarray(live_p1, dtype=np.float64)
    key_rho = None if probe._last_key_rho is None else np.asarray(probe._last_key_rho, dtype=np.float64)
    scores, score_addr, meta = decision_scores(probe, live_p1, key_rho=key_rho)
    winner = unique_winner(scores)
    gap = ranking_margin(scores, winner) if winner else 0.0
    handle = want if want is not None else winner
    gamma = probe_geometric_margin(probe, score_addr, scores, handle) if handle else 0.0
    raw = diagnostic_raw_live_scores(probe, live_p1)
    return {
        "scores": {k: float(v) for k, v in scores.items()},
        "winner": winner,
        "pairwise_score_gap": float(gap),
        "normalized_geometric_margin": float(gamma),
        "live_p1": live_p1.copy(),
        "live_p1_hash": arr_sha(live_p1),
        "scoring_address": np.asarray(score_addr, dtype=np.float64).copy(),
        "scoring_address_hash": arr_sha(score_addr),
        "key_rho": None if key_rho is None else key_rho.copy(),
        "retrieval_path": meta.get("path"),
        "retrieved_slot": meta.get("slot"),
        "familiar": meta.get("familiar"),
        "d1": meta.get("nearest_dist"),
        "d2": meta.get("second_nearest_dist"),
        "R": meta.get("R"),
        "recall_meta": dict(meta),
        "n_episodes": len(probe._episodes),
        "raw_live_diagnostic": raw,
    }


def perturb_live_p1_then_canonical(
    ag: NeuralCortex,
    live_p1: np.ndarray,
    key_rho: np.ndarray | None,
    want: str,
    *,
    domain: str,
    key: str,
    sigma: float,
    n: int,
    need: int,
) -> dict[str, Any]:
    """Perturb live P1, then rerun retrieval and canonical scoring."""
    rng = np.random.default_rng(domain_seed(domain, key))
    r0 = np.asarray(live_p1, dtype=np.float64).reshape(-1)
    nrm = float(np.linalg.norm(r0)) + 1e-12
    r_hat = r0 / nrm
    kr = None if key_rho is None else np.asarray(key_rho, dtype=np.float64)
    n_ok = 0
    trials: list[dict[str, Any]] = []
    for i in range(int(n)):
        unit = r_hat + rng.normal(0.0, float(sigma), size=r_hat.shape)
        pn = float(np.linalg.norm(unit)) + 1e-12
        unit = unit / pn
        scores, addr, meta = decision_scores(ag, unit, key_rho=kr)
        ok = unique_winner(scores) == want
        n_ok += int(ok)
        trials.append(
            {
                "trial": int(i),
                "winner": unique_winner(scores),
                "ranking_ok": bool(ok),
                "retrieval_path": meta.get("path"),
                "retrieved_slot": meta.get("slot"),
                "live_p1_hash": arr_sha(unit),
                "scoring_address_hash": arr_sha(addr),
            }
        )
    return {
        "n_ok": int(n_ok),
        "n": int(n),
        "stable": bool(n_ok >= int(need)),
        "perturbed": "live_p1",
        "then": "actuator_decision_scores",
        "trials": trials,
    }


def canonical_probe_map(
    ag: NeuralCortex,
    world: dict[str, Any],
    pairs: list[tuple[str, str]],
    *,
    tag: str,
    domain: str,
    geometric_min: float,
    sigma: float,
    perturb_n: int,
    perturb_need: int,
) -> dict[str, Any]:
    probes = []
    ranking_ok = True
    n_probe_correct = 0
    gammas: list[float] = []
    pert_ok = True
    for i, (cue, handle) in enumerate(pairs):
        live = canonical_cue_probe(ag, world, cue, tag=f"{tag}_p{i}", want=handle)
        rank = bool(live["winner"] == handle)
        ranking_ok = ranking_ok and rank
        n_probe_correct += int(rank)
        g = float(live["normalized_geometric_margin"])
        gammas.append(g)
        stab = perturb_live_p1_then_canonical(
            ag,
            live["live_p1"],
            live["key_rho"],
            handle,
            domain=domain,
            key=f"{tag}_{cue}",
            sigma=sigma,
            n=perturb_n,
            need=perturb_need,
        )
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
                "live_p1_hash": live["live_p1_hash"],
                "scoring_address_hash": live["scoring_address_hash"],
                "retrieval_path": live["retrieval_path"],
                "retrieved_slot": live["retrieved_slot"],
                "familiar": live["familiar"],
                "d1": live["d1"],
                "R": live["R"],
                "raw_live_winner_diagnostic": live["raw_live_diagnostic"]["winner"],
            }
        )
    min_g = min(gammas) if gammas else 0.0
    return {
        "probes": probes,
        "ranking_ok": bool(ranking_ok),
        "n_probe_correct": int(n_probe_correct),
        "min_normalized_geometric_margin": float(min_g),
        "geometric_ok": bool(min_g >= float(geometric_min)),
        "perturbation_ok": bool(pert_ok),
        "behavioral_scorer": "actuator_decision_scores",
    }


def motor_loop_uses_decision_scores() -> bool:
    src = NEURAL.read_text(encoding="utf-8")
    return "act_scores, _score_addr, _recall_meta = self.actuator_decision_scores(addr)" in src


def _calls_actuator_scores_outside_diagnostic(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bad: list[str] = []

    class V(ast.NodeVisitor):
        def __init__(self) -> None:
            self.fn: list[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.fn.append(node.name)
            self.generic_visit(node)
            self.fn.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "actuator_scores":
                cur = self.fn[-1] if self.fn else ""
                if not (path.name == "canonical_act_probe.py" and cur == "diagnostic_raw_live_scores"):
                    bad.append(f"{path.name}:{node.lineno}:{cur or '<module>'}")
            self.generic_visit(node)

    V().visit(tree)
    return bad


def refuse_raw_behavioral_actuator_scores(paths: tuple[Path, ...] | None = None) -> list[str]:
    hits: list[str] = []
    for p in paths or BEHAVIORAL_RUNNERS:
        if p.exists():
            hits.extend(_calls_actuator_scores_outside_diagnostic(p))
    return hits


def _self_check() -> None:
    assert inspect.signature(NeuralCortex.actuator_decision_scores)
    assert motor_loop_uses_decision_scores()
