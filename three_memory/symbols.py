"""Native codes for v7. Not English. Not DNA letters.

Observations are 0/1 bits (see Obs.vector). Notes on disk are integer tags.
0–3 for actions match env.Action. Door 0=red, 1=blue. Experimenter legend only.
"""

from __future__ import annotations

from typing import Any

# Action codes == env.Action
ACT_WAIT = 0
ACT_OPEN = 1
ACT_USE_KEY = 2
ACT_PICK_KEY = 3

DOOR_RED = 0
DOOR_BLUE = 1
DOOR_GREEN = 2
REQ_KEY = 1

RED_FACT_ID = "d0"
BLUE_FACT_ID = "d1"
GREEN_FACT_ID = "d2"


def encode_tags(tags: dict[str, Any]) -> str:
    return ";".join(f"{k}={v}" for k, v in sorted(tags.items()))


def record_to_tagfile(fact_id: str, tags: dict[str, Any]) -> str:
    lines = [f"# {fact_id}"]
    for k, v in sorted(tags.items()):
        lines.append(f"{k}={v}")
    return "\n".join(lines) + "\n"


def parse_tagfile(text: str) -> tuple[str, dict[str, Any]]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    fact_id = "x"
    tags: dict[str, Any] = {}
    if lines and lines[0].startswith("#"):
        fact_id = lines[0][1:].strip() or "x"
        body = lines[1:]
    else:
        body = lines
    for ln in body:
        if "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        k, v = k.strip(), v.strip()
        try:
            tags[k] = int(v)
        except ValueError:
            tags[k] = v
    return fact_id, tags
