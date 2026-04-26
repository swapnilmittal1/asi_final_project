"""
Shared intervention selection for experiment runners.

For the **main matrix**, greedy oracle ``n_runs`` matches final evaluation ``n_runs`` so
marginal-gain estimates use the same Monte Carlo budget as reported metrics.
"""

from __future__ import annotations

from typing import Dict, List, Set

import networkx as nx

from src.models.interventions import (
    select_fairness_aware_greedy_blocking,
    select_greedy_blocking,
    select_no_intervention,
    select_pagerank_blocking,
    select_random_blocking,
    select_top_degree_blocking,
)


def compute_method_blocked_sets(
    graph: nx.Graph,
    seeds: List[int],
    budget: int,
    p: float,
    communities: Dict[int, int],
    *,
    n_runs_greedy: int,
    random_block_seed: int,
    greedy_seed: int,
    fairness_lambda: float,
    pool_top_n: int,
) -> Dict[str, Set[int]]:
    """
    Return blocked-node sets for all baseline and greedy methods.

    ``n_runs_greedy`` is passed to both greedy variants as the IC oracle budget (should match
    final evaluation ``n_runs`` in the main matrix for interpretability).
    """
    seed_set = set(seeds)
    out: Dict[str, Set[int]] = {}
    out["no_intervention"] = select_no_intervention(graph, 0)
    out["random_blocking"] = select_random_blocking(
        graph, budget, random_seed=random_block_seed, exclude=seed_set
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
    )
    out["fairness_aware_greedy_blocking"] = select_fairness_aware_greedy_blocking(
        graph,
        seeds,
        budget,
        communities=communities,
        lambda_fair=fairness_lambda,
        p=p,
        n_runs=n_runs_greedy,
        random_seed=greedy_seed + 1_000_003,
        candidate_pool_top_n=pool_top_n,
    )
    return out
