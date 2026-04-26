"""
Intervention node selection for simulated influence blocking.

Selections are structural (graph-based), not demographic targeting.
"""

from __future__ import annotations

import logging
import math
from typing import Iterable, List, Literal, Mapping, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np

from src.models.diffusion import simulate_ic

logger = logging.getLogger(__name__)

PAGERANK_ALPHA: float = 0.85
PAGERANK_MAX_ITER: int = 100
DEFAULT_GREEDY_POOL_TOP_N: int = 25
ObjectiveMetric = Literal["infected", "harm"]
CommunityPenaltyMetric = Literal[
    "none",
    "worst_community_infection_rate",
    "worst_community_harm_rate",
]


def sanitize_budget(
    graph: nx.Graph,
    budget: int,
    *,
    exclude: Optional[Iterable[int]] = None,
) -> int:
    """
    Clamp budget to the number of selectable nodes (optionally excluding a set).
    """
    if budget < 0:
        raise ValueError("budget must be non-negative")
    ex = {int(x) for x in (exclude or [])}
    n_cand = sum(1 for n in graph.nodes if int(n) not in ex)
    return min(int(budget), n_cand)


def sanitize_candidate_nodes(
    graph: nx.Graph,
    *,
    exclude: Optional[Iterable[int]] = None,
) -> List[int]:
    """Sorted list of graph nodes not in ``exclude``."""
    ex = {int(x) for x in (exclude or [])}
    return sorted(int(n) for n in graph.nodes if int(n) not in ex)


def select_no_intervention(
    _graph: nx.Graph,
    _budget: int = 0,
    *,
    _exclude: Optional[Iterable[int]] = None,
) -> Set[int]:
    """Return an empty intervention set (arguments kept for a uniform caller API)."""
    return set()


def select_random_blocking(
    graph: nx.Graph,
    budget: int,
    *,
    random_seed: int = 0,
    rng: Optional[np.random.Generator] = None,
    exclude: Optional[Iterable[int]] = None,
) -> Set[int]:
    """
    Uniformly sample ``budget`` distinct nodes without replacement.

    If ``rng`` is provided, ``random_seed`` is ignored.
    """
    candidates = sanitize_candidate_nodes(graph, exclude=exclude)
    k = sanitize_budget(graph, budget, exclude=exclude)
    if k == 0:
        return set()
    gen = rng if rng is not None else np.random.default_rng(int(random_seed))
    pick = gen.choice(len(candidates), size=k, replace=False)
    return {candidates[int(i)] for i in pick}


def select_top_degree_blocking(
    graph: nx.Graph,
    budget: int,
    *,
    exclude: Optional[Iterable[int]] = None,
) -> Set[int]:
    """
    Select nodes with largest degree; ties broken by ascending node id.
    """
    candidates = sanitize_candidate_nodes(graph, exclude=exclude)
    k = sanitize_budget(graph, budget, exclude=exclude)
    scored = [(int(n), int(graph.degree(n))) for n in candidates]
    scored.sort(key=lambda t: (-t[1], t[0]))
    return {scored[i][0] for i in range(k)}


def select_pagerank_blocking(
    graph: nx.Graph,
    budget: int,
    *,
    exclude: Optional[Iterable[int]] = None,
    alpha: float = PAGERANK_ALPHA,
) -> Set[int]:
    """
    Select nodes with largest PageRank; ties broken by ascending node id.

    Uses fixed damping ``alpha`` (default 0.85) for reproducibility across NetworkX versions.
    """
    candidates = sanitize_candidate_nodes(graph, exclude=exclude)
    k = sanitize_budget(graph, budget, exclude=exclude)
    if k == 0:
        return set()
    pr = nx.pagerank(graph, alpha=alpha, max_iter=PAGERANK_MAX_ITER)
    scored = [(int(n), float(pr[n])) for n in candidates]
    scored.sort(key=lambda t: (-t[1], t[0]))
    return {scored[i][0] for i in range(k)}


def build_default_greedy_candidate_pool(
    graph: nx.Graph,
    seeds: Iterable[int],
    top_n: int = DEFAULT_GREEDY_POOL_TOP_N,
    *,
    use_degree: bool = True,
    use_pagerank: bool = True,
    pagerank_alpha: float = PAGERANK_ALPHA,
) -> List[int]:
    """
    Restricted candidate pool: union of top-``top_n`` by degree and top-``top_n`` by PageRank.

    Excludes seed nodes. Documented default to keep greedy ``simulate_ic`` oracle calls bounded.
    """
    seed_set = {int(s) for s in seeds}
    nodes = [int(n) for n in graph.nodes if int(n) not in seed_set]
    if not nodes:
        return []
    picked: Set[int] = set()
    if use_degree:
        by_deg = sorted(nodes, key=lambda n: (-graph.degree(n), n))[:top_n]
        picked.update(by_deg)
    if use_pagerank:
        pr = nx.pagerank(graph, alpha=pagerank_alpha, max_iter=PAGERANK_MAX_ITER)
        by_pr = sorted(nodes, key=lambda n: (-pr[n], n))[:top_n]
        picked.update(by_pr)
    return sorted(picked)


def _greedy_marginal_scores(
    graph: nx.Graph,
    seeds: Sequence[int],
    budget: int,
    candidate_pool: Sequence[int],
    p: float,
    n_runs: int,
    communities: Optional[Mapping[int, int]],
    random_seed: Optional[int],
    primary_metric: ObjectiveMetric,
    community_penalty_metric: CommunityPenaltyMetric,
    community_tradeoff: float,
    scenario_harm_weight: float,
    node_harm_weights: Optional[Mapping[int, float]],
    propagation_multiplier: float,
) -> Tuple[Set[int], List[Tuple[int, float]]]:
    """
    Shared greedy loop for infection- and harm-aware objectives.

    Examples
    --------
    - Binary greedy: primary metric = ``infected``, no community penalty.
    - Fairness-aware greedy: ``infected`` + lambda * worst-community infection rate.
    - Harm-aware resilience greedy: ``harm`` + lambda * worst-community harm rate.

    Returns blocked set and per-step (chosen_node, objective_at_choice) log.
    """
    if community_penalty_metric != "none":
        if communities is None:
            raise ValueError(
                "Community-aware greedy requires a structural ``communities`` map "
                "(circles or detected partition)."
            )
        if community_tradeoff < 0:
            raise ValueError("community_tradeoff must be non-negative")

    seed_set = {int(s) for s in seeds}
    budget_eff = sanitize_budget(graph, budget, exclude=seed_set)
    pool = sorted({int(x) for x in candidate_pool if int(x) not in seed_set})
    if not pool or budget_eff == 0:
        return set(), []

    blocked: Set[int] = set()
    base_seed = 0 if random_seed is None else int(random_seed)
    trace: List[Tuple[int, float]] = []

    for step in range(budget_eff):
        best_v: Optional[int] = None
        best_score = float("inf")
        for v in pool:
            if v in blocked:
                continue
            trial = blocked | {v}
            res = simulate_ic(
                graph,
                seeds,
                blocked_nodes=trial,
                p=p,
                n_runs=n_runs,
                communities=communities,
                random_seed=base_seed,
                scenario_harm_weight=scenario_harm_weight,
                node_harm_weights=node_harm_weights,
                propagation_multiplier=propagation_multiplier,
            )
            if primary_metric == "infected":
                score = float(res["mean_infected"])
            elif primary_metric == "harm":
                score = float(res["mean_total_harm"])
            else:
                raise ValueError(f"Unknown primary metric: {primary_metric}")
            if community_penalty_metric == "worst_community_infection_rate":
                penalty = res["mean_worst_community_infection_rate"]
                if penalty is None:
                    raise RuntimeError("simulate_ic did not return worst-community infection metric")
                score += float(community_tradeoff) * float(penalty)
            elif community_penalty_metric == "worst_community_harm_rate":
                penalty = res["mean_worst_community_harm_rate"]
                if penalty is None:
                    raise RuntimeError("simulate_ic did not return worst-community harm metric")
                score += float(community_tradeoff) * float(penalty)
            if score < best_score or (math.isclose(score, best_score) and (best_v is None or v < best_v)):
                best_score = score
                best_v = v
        if best_v is None:
            break
        blocked.add(best_v)
        trace.append((best_v, best_score))
    return blocked, trace


def select_greedy_blocking(
    graph: nx.Graph,
    seeds: Sequence[int],
    budget: int,
    *,
    candidate_nodes: Optional[Sequence[int]] = None,
    p: float,
    n_runs: int,
    communities: Optional[Mapping[int, int]] = None,
    random_seed: Optional[int] = None,
    candidate_pool_top_n: int = DEFAULT_GREEDY_POOL_TOP_N,
    use_degree_pool: bool = True,
    use_pagerank_pool: bool = True,
    scenario_harm_weight: float = 1.0,
    node_harm_weights: Optional[Mapping[int, float]] = None,
    propagation_multiplier: float = 1.0,
) -> Set[int]:
    """
    Greedy blocking: repeatedly add the node minimizing estimated mean cascade size.

    Candidate pool defaults to the union of top-``candidate_pool_top_n`` nodes by degree and by
    PageRank (excluding seeds). Pass ``candidate_nodes`` to override the pool explicitly.
    """
    pool: List[int]
    if candidate_nodes is not None:
        pool = list(candidate_nodes)
    else:
        pool = build_default_greedy_candidate_pool(
            graph,
            seeds,
            top_n=candidate_pool_top_n,
            use_degree=use_degree_pool,
            use_pagerank=use_pagerank_pool,
        )
    blocked, _ = _greedy_marginal_scores(
        graph,
        seeds,
        budget,
        pool,
        p,
        n_runs,
        communities,
        random_seed,
        primary_metric="infected",
        community_penalty_metric="none",
        community_tradeoff=0.0,
        scenario_harm_weight=scenario_harm_weight,
        node_harm_weights=node_harm_weights,
        propagation_multiplier=propagation_multiplier,
    )
    return blocked


def select_fairness_aware_greedy_blocking(
    graph: nx.Graph,
    seeds: Sequence[int],
    budget: int,
    *,
    communities: Mapping[int, int],
    lambda_fair: float = 3.0,
    candidate_nodes: Optional[Sequence[int]] = None,
    p: float,
    n_runs: int,
    random_seed: Optional[int] = None,
    candidate_pool_top_n: int = DEFAULT_GREEDY_POOL_TOP_N,
    use_degree_pool: bool = True,
    use_pagerank_pool: bool = True,
    scenario_harm_weight: float = 1.0,
    node_harm_weights: Optional[Mapping[int, float]] = None,
    propagation_multiplier: float = 1.0,
) -> Set[int]:
    """
    Greedy blocking minimizing ``mean_infected + lambda_fair * mean_worst_community_infection_rate``.

    ``communities`` must map each graph node to a **structural** community id (int).
    """
    pool: List[int]
    if candidate_nodes is not None:
        pool = list(candidate_nodes)
    else:
        pool = build_default_greedy_candidate_pool(
            graph,
            seeds,
            top_n=candidate_pool_top_n,
            use_degree=use_degree_pool,
            use_pagerank=use_pagerank_pool,
        )
    blocked, _ = _greedy_marginal_scores(
        graph,
        seeds,
        budget,
        pool,
        p,
        n_runs,
        communities,
        random_seed,
        primary_metric="infected",
        community_penalty_metric="worst_community_infection_rate",
        community_tradeoff=lambda_fair,
        scenario_harm_weight=scenario_harm_weight,
        node_harm_weights=node_harm_weights,
        propagation_multiplier=propagation_multiplier,
    )
    return blocked


def select_harm_aware_greedy_blocking(
    graph: nx.Graph,
    seeds: Sequence[int],
    budget: int,
    *,
    candidate_nodes: Optional[Sequence[int]] = None,
    p: float,
    n_runs: int,
    communities: Optional[Mapping[int, int]] = None,
    random_seed: Optional[int] = None,
    candidate_pool_top_n: int = DEFAULT_GREEDY_POOL_TOP_N,
    use_degree_pool: bool = True,
    use_pagerank_pool: bool = True,
    scenario_harm_weight: float = 1.0,
    node_harm_weights: Optional[Mapping[int, float]] = None,
    propagation_multiplier: float = 1.0,
) -> Set[int]:
    """
    Greedy blocking minimizing expected total misinformation harm.
    """
    pool = (
        list(candidate_nodes)
        if candidate_nodes is not None
        else build_default_greedy_candidate_pool(
            graph,
            seeds,
            top_n=candidate_pool_top_n,
            use_degree=use_degree_pool,
            use_pagerank=use_pagerank_pool,
        )
    )
    blocked, _ = _greedy_marginal_scores(
        graph,
        seeds,
        budget,
        pool,
        p,
        n_runs,
        communities,
        random_seed,
        primary_metric="harm",
        community_penalty_metric="none",
        community_tradeoff=0.0,
        scenario_harm_weight=scenario_harm_weight,
        node_harm_weights=node_harm_weights,
        propagation_multiplier=propagation_multiplier,
    )
    return blocked


def select_harm_aware_resilience_greedy_blocking(
    graph: nx.Graph,
    seeds: Sequence[int],
    budget: int,
    *,
    communities: Mapping[int, int],
    lambda_resilience: float = 3.0,
    candidate_nodes: Optional[Sequence[int]] = None,
    p: float,
    n_runs: int,
    random_seed: Optional[int] = None,
    candidate_pool_top_n: int = DEFAULT_GREEDY_POOL_TOP_N,
    use_degree_pool: bool = True,
    use_pagerank_pool: bool = True,
    scenario_harm_weight: float = 1.0,
    node_harm_weights: Optional[Mapping[int, float]] = None,
    propagation_multiplier: float = 1.0,
) -> Set[int]:
    """
    Harm-aware community-resilience greedy objective.

    Objective:
    ``mean_total_harm + lambda_resilience * mean_worst_community_harm_rate``.
    """
    pool = (
        list(candidate_nodes)
        if candidate_nodes is not None
        else build_default_greedy_candidate_pool(
            graph,
            seeds,
            top_n=candidate_pool_top_n,
            use_degree=use_degree_pool,
            use_pagerank=use_pagerank_pool,
        )
    )
    blocked, _ = _greedy_marginal_scores(
        graph,
        seeds,
        budget,
        pool,
        p,
        n_runs,
        communities,
        random_seed,
        primary_metric="harm",
        community_penalty_metric="worst_community_harm_rate",
        community_tradeoff=lambda_resilience,
        scenario_harm_weight=scenario_harm_weight,
        node_harm_weights=node_harm_weights,
        propagation_multiplier=propagation_multiplier,
    )
    return blocked
