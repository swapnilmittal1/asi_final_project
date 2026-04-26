"""
Build JSON scenario configs that bind graph id, seeds, IC parameters, and budgets.

CoAID engagement informs **labels and descriptive quantiles** only; ``ic_propagation_p`` is a
simulator knob (see ``docs/experiment_plan.md``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.config import COAID_ARTICLES_ENGAGEMENT_CSV, DATA_PROCESSED, SCENARIOS_DIR

logger = logging.getLogger(__name__)

DEFAULT_BUDGETS: List[int] = [5, 10, 20, 40]


def _load_dev_ego_id() -> int:
    path = DATA_PROCESSED / "twitter_dev_ego_id.txt"
    if not path.exists():
        raise FileNotFoundError("Run build_twitter_graphs.py first (missing twitter_dev_ego_id.txt).")
    return int(path.read_text(encoding="utf-8").strip())


def _load_dev_community_source() -> str:
    meta_path = DATA_PROCESSED / "twitter_dev_metadata.json"
    if not meta_path.exists():
        return "circles"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return "detected_fallback" if meta.get("circles_fallback_needed") else "circles"


def _coaid_fake_engagement_summary(df: pd.DataFrame) -> Dict[str, Any]:
    fake = df.loc[df["label"] == 1, "engagement_total"].astype(float)
    log_eng = np.log1p(fake)
    q33, q66 = float(np.quantile(log_eng, 1 / 3)), float(np.quantile(log_eng, 2 / 3))
    return {
        "n_fake_articles": int(len(fake)),
        "engagement_total_min": int(fake.min()) if len(fake) else 0,
        "engagement_total_max": int(fake.max()) if len(fake) else 0,
        "log1p_tertile_cutoffs": [q33, q66],
        "note": (
            "Tertiles computed on log1p(engagement_total) for fake-labeled CoAID rows only; "
            "used to name low/medium/high virality scenarios, not to estimate IC parameters."
        ),
    }


def _scenario_template(
    *,
    scenario_id: str,
    ego_id: int,
    seed_strategy: str,
    n_seeds: int,
    propagation_regime: str,
    ic_propagation_p: float,
    community_source: str,
    coaid_summary: Dict[str, Any],
    extra_notes: str,
) -> Dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "ego_id": ego_id,
        "graph_edge_list": "data/processed/twitter_dev_edges.csv",
        "node_community_csv": "data/processed/twitter_dev_node_communities.csv",
        "seed_strategy": seed_strategy,
        "n_seeds": n_seeds,
        "propagation_regime": propagation_regime,
        "ic_propagation_p": ic_propagation_p,
        "intervention_budgets": DEFAULT_BUDGETS,
        "community_source": community_source,
        "coaid_engagement_summary": coaid_summary,
        "disclaimer": (
            "ic_propagation_p is a simulation setting for stress-testing methods; it is not a "
            "causal estimate of retweet probability derived from CoAID engagement."
        ),
        "extra_notes": extra_notes,
    }


def build_scenarios() -> List[Dict[str, Any]]:
    if not COAID_ARTICLES_ENGAGEMENT_CSV.exists():
        raise FileNotFoundError(
            f"Missing {COAID_ARTICLES_ENGAGEMENT_CSV}; run build_coaid_tables.py first."
        )
    articles = pd.read_csv(COAID_ARTICLES_ENGAGEMENT_CSV)
    coaid_summary = _coaid_fake_engagement_summary(articles)
    ego_id = _load_dev_ego_id()
    community_source = _load_dev_community_source()

    # Fix path note: actual file is twitter_dev_node_communities.csv — template updated below
    scenarios = [
        _scenario_template(
            scenario_id="low_virality_random",
            ego_id=ego_id,
            seed_strategy="random",
            n_seeds=5,
            propagation_regime="low",
            ic_propagation_p=0.01,
            community_source=community_source,
            coaid_summary=coaid_summary,
            extra_notes="Motivated by lower engagement mass in CoAID fake-news distribution; uses small p and few seeds.",
        ),
        _scenario_template(
            scenario_id="medium_virality_high_degree",
            ego_id=ego_id,
            seed_strategy="high_degree",
            n_seeds=10,
            propagation_regime="medium",
            ic_propagation_p=0.03,
            community_source=community_source,
            coaid_summary=coaid_summary,
            extra_notes="Default stress level; seeds on high-degree nodes.",
        ),
        _scenario_template(
            scenario_id="high_virality_concentrated",
            ego_id=ego_id,
            seed_strategy="community_concentrated",
            n_seeds=15,
            propagation_regime="high",
            ic_propagation_p=0.05,
            community_source=community_source,
            coaid_summary=coaid_summary,
            extra_notes="Higher p and more seeds concentrated in one community (handled at simulation time).",
        ),
    ]
    return scenarios


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = build_scenarios()
    for s in scenarios:
        out = SCENARIOS_DIR / f"{s['scenario_id']}.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump(s, f, indent=2)
        logger.info("Wrote %s", out)
    manifest = {"scenarios": [x["scenario_id"] for x in scenarios], "ego_id": scenarios[0]["ego_id"]}
    with (SCENARIOS_DIR / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote %s", SCENARIOS_DIR / "manifest.json")


if __name__ == "__main__":
    main()
