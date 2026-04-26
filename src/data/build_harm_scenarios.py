"""
Build severity-aware scenario JSONs for the harm extension.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

import pandas as pd

from src.config import COAID_SEVERITY_REGIMES_CSV, HARM_SCENARIOS_DIR

logger = logging.getLogger(__name__)

DEFAULT_PROPAGATION_MODES = ["constant", "bucket", "linear"]


def build_harm_scenarios() -> List[Dict[str, Any]]:
    if not COAID_SEVERITY_REGIMES_CSV.exists():
        raise FileNotFoundError(
            f"Missing {COAID_SEVERITY_REGIMES_CSV}; run python -m src.data.build_severity_tables first."
        )
    regimes = pd.read_csv(COAID_SEVERITY_REGIMES_CSV)
    rows: List[Dict[str, Any]] = []
    for _, regime in regimes.iterrows():
        for propagation_mode in DEFAULT_PROPAGATION_MODES:
            scenario_id = f"{regime['severity_regime']}_severity_{propagation_mode}_propagation"
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "severity_mode": "severity_aware",
                    "severity_regime": str(regime["severity_regime"]),
                    "severity_score": float(regime["severity_score_median"]),
                    "severity_label": str(regime["severity_regime"]),
                    "base_harm_weight": float(regime["base_harm_weight"]),
                    "propagation_mode": propagation_mode,
                    "propagation_multiplier": float(regime[f"propagation_multiplier_{propagation_mode}"]),
                    "objective_modes_supported": ["binary_count", "harm_total", "harm_plus_resilience"],
                    "notes": (
                        "Semi-synthetic severity scenario derived from CoAID fake-article severity "
                        "bucket summaries. No user-level mapping to SNAP nodes."
                    ),
                }
            )
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    HARM_SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = build_harm_scenarios()
    for scenario in scenarios:
        path = HARM_SCENARIOS_DIR / f"{scenario['scenario_id']}.json"
        path.write_text(json.dumps(scenario, indent=2), encoding="utf-8")
        logger.info("Wrote %s", path)
    manifest = {"scenarios": [row["scenario_id"] for row in scenarios]}
    (HARM_SCENARIOS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote %s", HARM_SCENARIOS_DIR / "manifest.json")


if __name__ == "__main__":
    main()
