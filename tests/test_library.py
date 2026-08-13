"""Select among many notes; collect from unread W."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from three_memory.byte_lm import LMConfig, TinyByteLM
from three_memory.bytes_util import PROBE
from three_memory.library import (
    DISTRACTOR_MARKERS,
    WorldLibrary,
    all_library_notes,
    write_notes,
)
from three_memory.lm_agent import LanguageAgent
from three_memory.md_store import MarkdownStore


def _model(device):
    torch.manual_seed(1)
    return TinyByteLM(LMConfig(n_embd=16, n_hidden=16)).to(device).eval()


def test_select_picks_longest_prefix(tmp_path: Path):
    write_notes(tmp_path, all_library_notes(include_love=True))
    device = torch.device("cpu")
    m = _model(device)
    a = LanguageAgent(
        m, device, retrieve_mode="note", retrieve_policy="select", store=MarkdownStore(tmp_path)
    )
    ctx = a._retrieve_context(PROBE)
    assert "NOTE: my lo -> v" in ctx
    assert "NOTE: lo ->" not in ctx
    assert "my lamp" not in ctx
    assert not any(marker in ctx for marker in DISTRACTOR_MARKERS)
    assert a.last_retrieve["n_rejected"] >= 10
    assert a.last_retrieve["n_chosen"] == 1


def test_dump_includes_clutter(tmp_path: Path):
    write_notes(tmp_path, all_library_notes(include_love=True))
    device = torch.device("cpu")
    m = _model(device)
    a = LanguageAgent(
        m, device, retrieve_mode="note", retrieve_policy="dump", store=MarkdownStore(tmp_path)
    )
    ctx = a._retrieve_context(PROBE)
    assert "torches" in ctx
    assert a.last_retrieve["n_chosen"] >= 10


def test_commit_copies_only_match(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_notes(w, all_library_notes(include_love=True))
    s.mkdir()
    device = torch.device("cpu")
    m = _model(device)
    a = LanguageAgent(
        m,
        device,
        retrieve_mode="note",
        collect_mode="commit",
        store=MarkdownStore(s),
        world=WorldLibrary(w),
    )
    a.probe(PROBE)
    files = list(MarkdownStore(s).list_files())
    assert files == ["my-lo.md"]
    assert (w / "my-lo.md").is_file()
    assert (w / "enter.md").is_file()


def test_peek_does_not_write_s(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_notes(w, all_library_notes(include_love=True))
    s.mkdir()
    device = torch.device("cpu")
    m = _model(device)
    a = LanguageAgent(
        m,
        device,
        retrieve_mode="note",
        collect_mode="peek",
        store=MarkdownStore(s),
        world=WorldLibrary(w),
    )
    ctx = a.probe(PROBE)["context"]
    assert "NOTE: my lo -> v" in ctx
    assert MarkdownStore(s).list_files() == []
    a.reset_rho()
    b = LanguageAgent(
        m, device, retrieve_mode="note", collect_mode="off", store=MarkdownStore(s), world=None
    )
    assert "NOTE:" not in b.probe(PROBE)["context"]


def test_collect_off_ignores_w(tmp_path: Path):
    w = tmp_path / "W"
    s = tmp_path / "S"
    write_notes(w, all_library_notes(include_love=True))
    s.mkdir()
    device = torch.device("cpu")
    m = _model(device)
    a = LanguageAgent(
        m,
        device,
        retrieve_mode="note",
        collect_mode="off",
        store=MarkdownStore(s),
        world=WorldLibrary(w),
    )
    assert "NOTE:" not in a.probe(PROBE)["context"]
    assert MarkdownStore(s).list_files() == []


if __name__ == "__main__":
    import tempfile

    for fn in (
        test_select_picks_longest_prefix,
        test_dump_includes_clutter,
        test_commit_copies_only_match,
        test_peek_does_not_write_s,
        test_collect_off_ignores_w,
    ):
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("ok")
