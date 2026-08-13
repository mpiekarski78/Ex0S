"""v2: retrieve stored snippets as ordinary text. No NOTE-copy in the prior."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_v1 import run_v1


def main() -> None:
    p = argparse.ArgumentParser(
        description="v2 raw retrieve (no taught NOTE-copy) vs BDH probes"
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(REPO_ROOT / "checkpoints" / "prior_plain.pt"),
    )
    p.add_argument("--exposures", type=int, default=8)
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()
    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        raise SystemExit(f"missing {ckpt}; run: python -m experiments.train_prior --plain")
    m = run_v1(
        ckpt,
        exposures=args.exposures,
        seed=args.seed,
        retrieve_mode="raw",
        run_prefix="v2",
    )
    print(json.dumps({k: m[k] for k in ("classification", "rationale", "run_dir", "retrieve_mode")}, indent=2))
    print("empty P(v)", m["empty_prior"]["p_v"], "P(r)", m["empty_prior"]["p_r"])
    print("S-on after reset P(v)", m["love_S_on_after_rho_reset"]["p_v"], "ctx:", repr(m["love_S_on_after_rho_reset"]["context"]))
    print("S-off after reset P(v)", m["love_S_off_after_rho_reset"]["p_v"])
    print("love fact", m["love_has_inspectable_fact"])
    print("NOTE in context?", "NOTE:" in m["love_S_on_after_rho_reset"]["context"])


if __name__ == "__main__":
    main()
