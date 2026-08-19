"""MEMLANG-1 sprint loop until 04:50 UTC. Discovery only. Product 0.0.004."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from random import Random

from experiments.run_memlang1_parallel import NEW_FAMILIES, run_cfgs, summarize
from three_memory.memlang.variants import variants_for

REPO = Path(__file__).resolve().parents[1]
PROV = REPO / "runs" / "memlang1" / "_provenance"
DEADLINE = datetime(2026, 8, 19, 4, 50, 22, tzinfo=timezone.utc)
START = datetime(2026, 8, 19, 0, 0, 22, tzinfo=timezone.utc)


def extra_mutants(n: int, *, salt: int) -> list[dict]:
    rng = Random(9000 + int(salt))
    out = []
    for i in range(int(n)):
        out.append(
            {
                "family": "mutant",
                "name": f"mutx_{salt}_{i:03d}",
                "scale": float(rng.choice([0.35, 0.7, 1.0, 1.5, 2.5])),
                "zca": bool(rng.choice([False, True, True])),
                "delay": float(rng.choice([0.0, 0.1, 0.2, 0.35])),
                "nudge": float(rng.choice([0.0, 0.0, 0.08, 0.15])),
                "eta": float(rng.choice([0.01, 0.03, 0.08])),
                "seed": 30000 + salt * 200 + i,
            }
        )
    return out


def checkpoint(tag: str, payload: dict) -> None:
    PROV.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = PROV / f"sprint_{tag}_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": str(path), "summary": payload.get("summary"), "selected": payload.get("selected")}, indent=2, default=str), flush=True)


def main() -> None:
    workers = 16
    all_hist: list[dict] = []
    selected = None
    wave = 0
    # Wave 6 structured
    cfgs = []
    for fam in ("tanh_rho", "delay_mix", "whiten_nudge", "mutant"):
        cfgs.extend(variants_for(fam))
    out = run_cfgs(cfgs, workers=workers, tag="wave6")
    all_hist.extend(out["history"])
    selected = out.get("selected") or selected
    checkpoint("wave6", {**out, "summary": summarize(out["history"])})
    if selected:
        print("STAGE_A_PASS", selected, flush=True)
        return
    salt = 0
    while datetime.now(timezone.utc) < DEADLINE:
        if selected:
            break
        salt += 1
        batch = extra_mutants(16, salt=salt)
        out = run_cfgs(batch, workers=workers, tag=f"mutant_batch_{salt}")
        all_hist.extend(out["history"])
        selected = out.get("selected") or selected
        checkpoint(
            f"batch{salt}",
            {
                **out,
                "summary": summarize(out["history"]),
                "cumulative": summarize(all_hist),
                "utc": datetime.now(timezone.utc).isoformat(),
                "families_catalog": NEW_FAMILIES,
            },
        )
        if selected:
            print("STAGE_A_PASS", selected, flush=True)
            break
    final = {
        "program": "MEMLANG-1",
        "cumulative": summarize(all_hist),
        "n_history": len(all_hist),
        "selected": selected,
        "utc": datetime.now(timezone.utc).isoformat(),
        "deadline": DEADLINE.isoformat(),
        "elapsed_from_program_start_s": (datetime.now(timezone.utc) - START).total_seconds(),
        "candidate_v41_lock": False,
        "install_W_star": False,
    }
    checkpoint("final", final)


if __name__ == "__main__":
    main()
