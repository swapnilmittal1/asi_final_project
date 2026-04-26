"""
Shared utilities for severity- and harm-aware experiment runners.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Set

import networkx as nx
import pandas as pd

from src.config import COAID_SEVERITY_REGIMES_CSV
from src.models.interventions import (
    select_fairness_aware_greedy_blocking,
    select_greedy_blocking,
    select_harm_aware_greedy_blocking,
    select_harm_aware_resilience_greedy_blocking,
    select_no_intervention,
    select_pagerank_blocking,
    select_random_blocking,
    select_top_degree_blocking,
)
from src.models.severity import SeverityProfile, make_profile_from_score

HARM_METHODS: List[str] = [
    "no_intervention",
    "random_blocking",
    "top_degree_blocking",
    "pagerank_blocking",
    "greedy_blocking",
    "fairness_aware_greedy_blocking",
    "harm_aware_greedy_blocking",
    "harm_aware_resilience_greedy_blocking",
]


@dataclass(frozen=True)
class HarmScenarioSpec:
    severity_regime: str
    propagation_mode: str
    profile: SeverityProfile


def load_severity_specs(propagation_mode: str = "constant") -> Dict[str, HarmScenarioSpec]:
    """
    Load regime-level severity profiles from the processed CoAID summary.
    """
    if not COAID_SEVERITY_REGIMES_CSV.exists():
        raise FileNotFoundError(
            f"Missing {COAID_SEVERITY_REGIMES_CSV}; run python -m src.data.build_severity_tables first."
        )
    df = pd.read_csv(COAID_SEVERITY_REGIMES_CSV)
    out: Dict[str, HarmScenarioSpec] = {}
    for _, row in df.iterrows():
        regime = str(row["severity_regime"])
        propagation_col = f"propagation_multiplier_{propagation_mode}"
        if propagation_col not in df.columns:
            raise ValueError(f"Severity regimes missing column: {propagation_col}")
        profile = make_profile_from_score(
            float(row["severity_score_median"]),
            construction_source=str(row.get("construction_source", "coaid_regime_table")),
            propagation_mode="constant",
            notes=str(row.get("severity_notes", "")),
        )
        profile = SeverityProfile(
            severity_score=profile.severity_score,
            severity_label=profile.severity_label,
            base_harm_weight=float(row["base_harm_weight"]),
            propagation_multiplier=float(row[propagation_col]),
            construction_source=profile.construction_source,
            notes=profile.notes,
        )
        out[regime] = HarmScenarioSpec(
            severity_regime=regime,
            propagation_mode=propagation_mode,
            profile=profile,
        )
    return out


def method_objective_family(method: str) -> str:
    """Human-readable optimization family for reporting."""
    if method in {"greedy_blocking", "fairness_aware_greedy_blocking"}:
        return "binary_objective"
    if method in {"harm_aware_greedy_blocking", "harm_aware_resilience_greedy_blocking"}:
        return "harm_objective"
    return "heuristic_baseline"


def compute_harm_method_blocked_sets(
    graph: nx.Graph,
    seeds: Sequence[int],
    budget: int,
    p: float,
    communities: Mapping[int, int],
    *,
    n_runs_greedy: int,
    random_block_seed: int,
    greedy_seed: int,
    fairness_lambda: float,
    resilience_lambda: float,
    pool_top_n: int,
    severity_profile: SeverityProfile,
) -> Dict[str, Set[int]]:
    """
    Return blocked-node sets for binary and harm-aware methods.
    """
    seed_set = set(int(x) for x in seeds)
    harm_kwargs = {
        "scenario_harm_weight": severity_profile.base_harm_weight,
        "propagation_multiplier": severity_profile.propagation_multiplier,
    }
    out: Dict[str, Set[int]] = {}
    out["no_intervention"] = select_no_intervention(graph, 0)
    out["random_blocking"] = select_random_blocking(
        graph,
        budget,
        random_seed=random_block_seed,
        exclude=seed_set,
    )
    out["top_degree_blocking"] = select_top_degree_blocking(graph, budget, exclude=seed_set)
    out["pagerank_blocking"] = select_pagerank_blocking(graph, budget, exclude=seed_set)
    out["greedy_blocking"] = select_greedy_blocking(
        graph,
        seeds,
        budget,
        p=p,
        n_runs=n_runs_greedy,
        communities=communities,
        random_seed=greedy_seed,
        candidate_pool_top_n=pool_top_n,
        **harm_kwargs,
    )
    out["fairness_aware_greedy_blocking"] = select_fairness_aware_greedy_blocking(
        graph,
        seeds,
        budget,
        communities=communities,
        lambda_fair=fairness_lambda,
        p=p,
        n_runs=n_runs_greedy,
        random_seed=greedy_seed + 100_003,
        candidate_pool_top_n=pool_top_n,
        **harm_kwargs,
    )
    out["harm_aware_greedy_blocking"] = select_harm_aware_greedy_blocking(
        graph,
        seeds,
        budget,
        p=p,
        n_runs=n_runs_greedy,
        communities=communities,
        random_seed=greedy_seed + 200_003,
        candidate_pool_top_n=pool_top_n,
        **harm_kwargs,
    )
    out["harm_aware_resilience_greedy_blocking"] = select_harm_aware_resilience_greedy_blocking(
        graph,
        seeds,
        budget,
        communities=communities,
        lambda_resilience=resilience_lambda,
        p=p,
        n_runs=n_runs_greedy,
        random_seed=greedy_seed + 300_003,
        candidate_pool_top_n=pool_top_n,
        **harm_kwargs,
    )
    return out
