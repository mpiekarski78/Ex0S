"""TM.0.16.RELATE: candidate relations under ambiguity battery + freeze locks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_tm011compose import make
from experiments.run_tm016relate import (
    CELL_IDS,
    GENOME_016_LOCK,
    PREREG_LOCK,
    PREREGISTERED_CLAIM,
    RELATE_LOCK,
    make_relate,
    run_relate,
    verify_genome_016,
    verify_prereg_lock,
    verify_relate_lock,
)
from three_memory.policy import UsePolicy


def test_prereg_intact():
    ok, why, lock = verify_prereg_lock()
    assert ok, why
    assert lock["observation_abi"] == "observe_event"
    assert lock["episode_boundary_api"] == "end_event_episode"
    assert lock["earned_next"] is False
    assert lock["ex0s"] is None
    assert lock["preregistered_claim"] == PREREGISTERED_CLAIM
    assert lock["candidate_rule"]["reads_focus"] is False
    assert lock["candidate_rule"]["prune_losers"] is False
    assert "put final mechanism/cell SHAs into this prereg lock" in lock["refuse"]
    assert "agent_sha" not in lock


def test_relate_battery_and_locks():
    summary = run_relate(seed=12345, write_locks=False)
    assert summary["ok"], summary
    assert summary["n_ok"] == summary["n_cells"] == 16
    assert summary["earned_next"] is False
    assert summary["ex0s"] is None
    assert summary["unexpected_writes"] == 0
    by = {r["cell"]: r for r in summary["rows"]}
    assert by["D0_birth_unreachable"]["ok"]
    assert by["D1_ambiguous_exposure_hold"]["evidence_tie"] is True
    assert by["D2_varying_clutter_winner"]["lived_bind"] == "y"
    assert by["D8_dual_strip"]["test_a_motor"] == "hold"
    assert by["D8_dual_strip"]["test_b_motor"] == "hold"
    assert by["D8_dual_strip"]["restore_motor"] == "press"
    assert by["D9_focus_not_relation_oracle"]["identical_skel"] is True
    assert by["D9_focus_not_relation_oracle"]["focus_leak_blocked"] is True
    assert by["D2_varying_clutter_winner"]["pruned"] is False
    assert "x->u:1" in by["D2_varying_clutter_winner"]["losers"]
    assert by["D10_irreducible_ambiguity"]["evidence_tie"] is True
    assert by["D11_episode_boundary"]["seam_without_boundary"] is True
    assert by["D11_episode_boundary"]["no_seam_with_boundary"] is True
    assert by["D15_nasty"]["empty_no_write"] is True
    assert by["D15_nasty"]["n1_legal_teach"] is True
    assert by["D15_nasty"]["no_harness_setattr"] is True

    ok, why, _ = verify_genome_016()
    assert ok, why
    ok2, why2, _ = verify_relate_lock(summary["rows"])
    assert ok2, why2
    lock = json.loads(RELATE_LOCK.read_text(encoding="utf-8"))
    assert lock["earned_next"] is False
    assert lock["ex0s"] is None
    assert lock["observation_abi"] == "observe_event"
    assert lock["reads_focus"] is False
    assert lock["cell_ids"] == list(CELL_IDS)
    g = json.loads(GENOME_016_LOCK.read_text(encoding="utf-8"))
    assert g["use_acquire_relate"] is True
    assert g["use_acquire_skel"] is True
    assert g["use_acquire_ctx"] is True
    assert g["earned_next"] is False
    assert g["reads_focus"] is False


def test_relate_off_by_default():
    from tempfile import TemporaryDirectory

    with TemporaryDirectory(prefix="tm016_off_") as tmp:
        s = Path(tmp) / "s"
        s.mkdir()
        ag = make(
            s,
            None,
            UsePolicy(seed=1),
            explore_epsilon=0.0,
            use_context_kappa=True,
            use_acquire_ctx=True,
            use_acquire_skel=True,
        )
        assert getattr(ag, "use_acquire_relate", False) is False
        out = ag.observe_event({"visible": ["x"], "focus": "x"})
        assert out.get("why") == "relate_off"
        ag2 = make_relate(s, UsePolicy(seed=1))
        assert ag2.use_acquire_relate is True


def test_make011compose_untouched():
    import inspect
    from experiments import run_tm011compose

    src = inspect.getsource(run_tm011compose.make)
    assert "use_acquire_relate" not in src
    assert "use_context_kappa" in src


def test_prereg_not_rewritten_with_freeze_shas():
    lock = json.loads(PREREG_LOCK.read_text(encoding="utf-8"))
    for banned in (
        "agent_sha",
        "make_relate_sha",
        "run_tm016relate_sha",
        "observe_event_sha",
        "cell_shas",
    ):
        assert banned not in lock


def test_focus_unused_in_observe_event_source():
    import ast
    import inspect
    import textwrap

    from three_memory.agent import ThreeMemoryAgent

    src = textwrap.dedent(inspect.getsource(ThreeMemoryAgent.observe_event))
    tree = ast.parse(src)
    focus_reads: list[str] = []

    class V(ast.NodeVisitor):
        def visit_Subscript(self, node: ast.Subscript) -> None:
            if isinstance(node.value, ast.Name) and node.value.id == "event":
                sl = node.slice
                if isinstance(sl, ast.Constant) and sl.value == "focus":
                    focus_reads.append("subscript")
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "event"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "focus"
            ):
                focus_reads.append("get")
            self.generic_visit(node)

    V().visit(tree)
    assert focus_reads == []


if __name__ == "__main__":
    test_prereg_intact()
    test_relate_battery_and_locks()
    test_relate_off_by_default()
    test_make011compose_untouched()
    test_prereg_not_rewritten_with_freeze_shas()
    test_focus_unused_in_observe_event_source()
    print("ok")
