"""v6 use-skill: tool grammar and fewshot demos, no probe facts in the skill."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from three_memory.byte_lm import LMConfig, TinyByteLM
from three_memory.bytes_util import BANNED, PROBE, V_ID
from three_memory.library import FEWSHOT_DEMOS, WorldLibrary, all_library_notes, write_notes
from three_memory.lm_agent import LanguageAgent
from three_memory.md_store import MarkdownStore


def _model(device):
    torch.manual_seed(1)
    return TinyByteLM(LMConfig(n_embd=16, n_hidden=16)).to(device).eval()


def test_fewshot_demos_have_no_probe_facts():
    for b in BANNED:
        assert b not in FEWSHOT_DEMOS


def test_tool_context_is_only_probe(tmp_path: Path):
    write_notes(tmp_path, all_library_notes(include_love=True))
    device = torch.device("cpu")
    a = LanguageAgent(
        _model(device),
        device,
        retrieve_mode="tool",
        store=MarkdownStore(tmp_path),
    )
    out = a.probe(PROBE)
    assert out["context"] == PROBE
    assert "NOTE:" not in out["context"]
    assert "love" not in out["context"]
    assert V_ID in out["retrieve"]["tool_next_ids"]
    assert out["p_v"] > 0.0


def test_tool_raises_p_v_vs_empty(tmp_path: Path):
    s = tmp_path / "S"
    empty = tmp_path / "empty"
    write_notes(s, all_library_notes(include_love=True))
    empty.mkdir()
    device = torch.device("cpu")
    m = _model(device)
    prior = LanguageAgent(m, device, retrieve_mode="tool", store=MarkdownStore(empty)).probe(PROBE)
    hit = LanguageAgent(m, device, retrieve_mode="tool", store=MarkdownStore(s)).probe(PROBE)
    assert hit["p_v"] > prior["p_v"]
    assert hit["argmax"] == V_ID or hit["p_v"] - prior["p_v"] > 0.05


def test_fewshot_puts_demos_and_fact(tmp_path: Path):
    write_notes(tmp_path, all_library_notes(include_love=True))
    device = torch.device("cpu")
    a = LanguageAgent(
        _model(device),
        device,
        retrieve_mode="fewshot",
        store=MarkdownStore(tmp_path),
    )
    ctx = a.probe(PROBE)["context"]
    assert FEWSHOT_DEMOS in ctx
    assert "NOTE: my lo -> v" in ctx
    assert ctx.endswith(PROBE)


def test_tool_commit_then_unmount(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_notes(w, all_library_notes(include_love=True))
    s.mkdir()
    device = torch.device("cpu")
    m = _model(device)
    a = LanguageAgent(
        m,
        device,
        retrieve_mode="tool",
        collect_mode="commit",
        store=MarkdownStore(s),
        world=WorldLibrary(w),
    )
    a.probe(PROBE)
    assert MarkdownStore(s).list_files() == ["my-lo.md"]
    b = LanguageAgent(m, device, retrieve_mode="tool", collect_mode="off", store=MarkdownStore(s), world=None)
    b.reset_rho()
    out = b.probe(PROBE)
    assert V_ID in out["retrieve"]["tool_next_ids"]
    assert out["context"] == PROBE


if __name__ == "__main__":
    import tempfile

    test_fewshot_demos_have_no_probe_facts()
    with tempfile.TemporaryDirectory() as d:
        test_tool_context_is_only_probe(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_tool_raises_p_v_vs_empty(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_fewshot_puts_demos_and_fact(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_tool_commit_then_unmount(Path(d))
    print("ok")
