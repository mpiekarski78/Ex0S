"""Summarize MEMLANG-1 telemetry by family and decision. Product 0.0.004."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

TELEMETRY = Path(__file__).resolve().parents[1] / "runs" / "memlang1"


def main() -> None:
    by_impl: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(TELEMETRY.glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("telemetry_schema") != "MEMLANG-1.TELEMETRY.v2":
            continue
        by_impl[str(rec.get("implementation_sha") or "?")].append(rec)
    out: dict[str, dict] = {}
    for impl, rows in by_impl.items():
        dec = Counter(str(r.get("decision_code")) for r in rows)
        fam = defaultdict(Counter)
        passed = [r for r in rows if r.get("decision_code") == "stage_a_integrated_pass"]
        for r in rows:
            fam[str(r.get("family"))][str(r.get("decision_code"))] += 1
        out[impl[:16]] = {
            "n": len(rows),
            "implementation_sha": impl,
            "by_decision": dict(dec),
            "by_family": {k: dict(v) for k, v in sorted(fam.items())},
            "n_pass": len(passed),
            "passes": [{"family": p.get("family"), "name": (p.get("config") or {}).get("name"), "run_id": p.get("run_id")} for p in passed],
        }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
