"""Markdown store: files on disk, reload, no embeddings."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from three_memory.byte_lm import LMConfig, TinyByteLM
from three_memory.bytes_util import PROBE
from three_memory.lm_agent import LanguageAgent
from three_memory.md_store import MarkdownStore
from three_memory.store import FactRecord


def test_write_and_reload(tmp_path: Path):
    store = MarkdownStore(tmp_path)
    rec = FactRecord(
        fact_id="s",
        what="my lo -> v",
        when=0,
        drive_scores={},
        tags={"prefix": "my lo", "next": "v", "snippet": "my love\n"},
    )
    assert store.write(rec)
    files = store.list_files()
    assert files == ["my-lo.md"]
    text = (tmp_path / "my-lo.md").read_text(encoding="utf-8")
    assert text.startswith("# my lo\n")
    assert "my love" in text
    assert "embedding" not in text.lower()

    other = MarkdownStore(tmp_path)
    assert len(other) == 1
    loaded = other.records()[0]
    assert loaded.tags["prefix"] == "my lo"
    assert loaded.tags["next"] == "v"
    assert loaded.tags["snippet"].startswith("my love")


def test_handwritten_md(tmp_path: Path):
    (tmp_path / "note.md").write_text("# my lo\n\nmy love\n", encoding="utf-8")
    store = MarkdownStore(tmp_path)
    rec = store.records()[0]
    assert rec.tags["prefix"] == "my lo"
    assert rec.tags["next"] == "v"


def test_reset_deletes_files(tmp_path: Path):
    store = MarkdownStore(tmp_path)
    store.write(
        FactRecord(
            fact_id="s",
            what="my lo -> v",
            when=0,
            drive_scores={},
            tags={"prefix": "my lo", "snippet": "my love\n"},
        )
    )
    assert (tmp_path / "my-lo.md").is_file()
    store.reset()
    assert list(tmp_path.glob("*.md")) == []
    assert len(store) == 0


def test_disabled_writes_no_files(tmp_path: Path):
    store = MarkdownStore(tmp_path, enabled=False)
    ok = store.write(
        FactRecord(
            fact_id="s",
            what="x",
            when=0,
            drive_scores={},
            tags={"prefix": "my lo", "snippet": "my love\n"},
        )
    )
    assert not ok
    assert list(tmp_path.glob("*.md")) == []


def test_agent_note_and_raw_from_md(tmp_path: Path):
    (tmp_path / "my-lo.md").write_text("# my lo\n\nmy love\n", encoding="utf-8")
    device = torch.device("cpu")
    torch.manual_seed(1)
    m = TinyByteLM(LMConfig(n_embd=16, n_hidden=16)).to(device).eval()
    note = LanguageAgent(
        m, device, retrieve_mode="note", store=MarkdownStore(tmp_path)
    )
    raw = LanguageAgent(
        m, device, retrieve_mode="raw", store=MarkdownStore(tmp_path)
    )
    assert "NOTE:" in note._retrieve_context(PROBE)
    assert "my lo -> v" in note._retrieve_context(PROBE)
    ctx = raw._retrieve_context(PROBE)
    assert ctx.startswith("my love")
    assert "NOTE:" not in ctx


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        test_write_and_reload(p)
    with tempfile.TemporaryDirectory() as d:
        test_handwritten_md(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_reset_deletes_files(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_disabled_writes_no_files(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_agent_note_and_raw_from_md(Path(d))
    print("ok")
