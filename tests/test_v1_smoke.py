"""v1 smoke: NOTE follow + store retrieve into context."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from three_memory.byte_lm import LMConfig, TinyByteLM, hash_lm, next_byte_logits, softmax
from three_memory.bytes_util import PROBE, encode_bytes
from three_memory.corpus import make_note_example, strip_probe_facts
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
    h1 = hash_lm(m)
    h2 = hash_lm(m)
    assert h1 == h2


def test_store_prefix_retrieve():
    device = torch.device("cpu")
    torch.manual_seed(1)
    m = TinyByteLM(LMConfig(n_embd=16, n_hidden=16)).to(device).eval()
    a = LanguageAgent(m, device, store_enabled=True)
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
    assert "my lo -> v" in notes


if __name__ == "__main__":
    test_strip_removes_probe_facts()
    test_hash_stable()
    test_store_prefix_retrieve()
    print("ok")
