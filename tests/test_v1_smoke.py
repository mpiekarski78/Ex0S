"""v1/v2 smoke: strip, hash, NOTE retrieve, raw retrieve."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from three_memory.byte_lm import LMConfig, TinyByteLM, hash_lm
from three_memory.bytes_util import PROBE
from three_memory.corpus import strip_probe_facts
from three_memory.lm_agent import LanguageAgent
from three_memory.store import FactRecord


def test_strip_removes_probe_facts():
    s = strip_probe_facts("my lord and my love and my lo")
    assert "lord" not in s
    assert "love" not in s
    assert "my lo" not in s


def test_hash_stable():
    torch.manual_seed(0)
    m = TinyByteLM(LMConfig(n_embd=16, n_hidden=16))
    assert hash_lm(m) == hash_lm(m)


def test_store_prefix_retrieve():
    device = torch.device("cpu")
    torch.manual_seed(1)
    m = TinyByteLM(LMConfig(n_embd=16, n_hidden=16)).to(device).eval()
    a = LanguageAgent(m, device, store_enabled=True, retrieve_mode="note")
    a.store.write(
        FactRecord(
            fact_id="x",
            what="my lo -> v",
            when=0,
            drive_scores={},
            tags={"prefix": PROBE, "next": "v"},
        )
    )
    notes = a._notes_for(PROBE)
    assert "NOTE:" in notes
    assert "my lo -> v" in notes


def test_raw_retrieve_replays_snippet_not_note():
    device = torch.device("cpu")
    torch.manual_seed(1)
    m = TinyByteLM(LMConfig(n_embd=16, n_hidden=16)).to(device).eval()
    a = LanguageAgent(m, device, store_enabled=True, retrieve_mode="raw")
    a.store.write(
        FactRecord(
            fact_id="s",
            what="my love",
            when=0,
            drive_scores={},
            tags={"snippet": "my love\n", "prefix": "my lo"},
        )
    )
    ctx = a._retrieve_context(PROBE)
    assert ctx.startswith("my love")
    assert "NOTE:" not in ctx


if __name__ == "__main__":
    test_strip_removes_probe_facts()
    test_hash_stable()
    test_store_prefix_retrieve()
    test_raw_retrieve_replays_snippet_not_note()
    print("ok")
