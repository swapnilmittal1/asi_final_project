"""
Write ``selected_egos.json`` and export processed artifacts for non-dev egos.

Run: ``python -m src.data.export_selected_egos`` from project root.
"""

from __future__ import annotations

import json
import logging

from src.config import DATA_PROCESSED, EGOS_PROCESSED_DIR, SELECTED_EGOS_JSON
from src.data.ego_artifacts import export_ego_artifacts

logger = logging.getLogger(__name__)

MANIFEST = {
    "description": "Dev ego uses legacy twitter_dev_* files; others live under data/processed/egos/<ego_id>/",
    "egos": [
        {
            "ego_id": 17723880,
            "label": "dev",
            "legacy_dev_paths": True,
            "matrix_role": "debugging",
        },
        {
            "ego_id": 17602896,
            "label": "medium_circles",
            "legacy_dev_paths": False,
            "notes": "Slightly larger than dev; non-empty SNAP circles.",
        },
        {
            "ego_id": 18534908,
            "label": "moderate_large_circles",
            "legacy_dev_paths": False,
            "notes": "More nodes/edges than dev; multiple circles.",
        },
        {
            "ego_id": 267383914,
            "label": "detected_communities",
            "legacy_dev_paths": False,
            "notes": "Empty .circles in SNAP; communities from greedy modularity.",
        },
        {
            "ego_id": 88639412,
            "label": "large_circles",
            "legacy_dev_paths": False,
            "notes": "Larger ego (~239 nodes); circles non-empty; for main conclusions.",
        },
    ],
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    SELECTED_EGOS_JSON.write_text(json.dumps(MANIFEST, indent=2), encoding="utf-8")
    logger.info("Wrote %s", SELECTED_EGOS_JSON)

    for entry in MANIFEST["egos"]:
        if entry.get("legacy_dev_paths"):
            continue
        eid = int(entry["ego_id"])
        export_ego_artifacts(eid, EGOS_PROCESSED_DIR / str(eid))


if __name__ == "__main__":
    main()
