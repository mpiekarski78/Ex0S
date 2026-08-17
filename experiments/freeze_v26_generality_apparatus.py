"""Freeze TM.0.23.CORTEX.GENERALITY.v26 runner. No score. No neural edit."""

from __future__ import annotations

import json

from experiments.cortex_v26_generality import freeze_generality_runner

if __name__ == "__main__":
    print(json.dumps(freeze_generality_runner(), indent=2))
