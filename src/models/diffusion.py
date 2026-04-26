"""
Independent Cascade (IC) diffusion on undirected graphs (simulation study).

Structural communities are graph-derived only; results do not imply real-world
causal contagion on Twitter.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np

from src.models.harm_metrics import community_harm_rates, total_harm

logger = logging.getLogger(__name__)

ICResult = Dict[str, Any]


def _normalize_nodes(graph: nx.Graph, nodes: Iterable[int]) -> List[int]:
    out: List[int] = []
    gnodes = set(graph.nodes)
    for x in nodes:
        xi = int(x)
        if xi not in gnodes:
            logger.debug("Skipping seed %s not in graph", xi)
            continue
        out.append(xi)
    return out


def _edge_activation_prob(
    u: int,
    v: int,
    p: float,
    edge_probs: Optional[nx.Graph],
    propagation_multiplier: float,
) -> float:
    if edge_probs is None:
        base_prob = float(p)
    else:
        data = edge_probs.get_edge_data(u, v)
        if data is None:
            raise ValueError(f"edge_probs missing edge ({u}, {v})")
        if "p" in data:
            base_prob = float(data["p"])
        elif "weight" in data:
            base_prob = float(data["weight"])
        else:
            raise ValueError(f"edge ({u}, {v}) needs 'p' or 'weight' in edge_probs")
    return min(1.0, max(0.0, base_prob * float(propagation_multiplier)))


def _ic_single_run(
    graph: nx.Graph,
    seed_nodes: Sequence[int],
    blocked: Set[int],
    p: float,
    rng: np.random.Generator,
    edge_probs: Optional[nx.Graph],
    max_steps: Optional[int],
    propagation_multiplier: float,
) -> Tuple[Set[int], int]:
    """
    One IC sample using synchronous waves.

    Returns
    -------
    infected, n_waves
    """
    nodes = graph.nodes
    infected: Set[int] = set()
    for s in sorted(seed_nodes):
        if s in blocked:
            continue
        if s in nodes:
            infected.add(s)

    if not infected:
        return infected, 0

    q = deque(sorted(infected))
    waves = 0
    while q:
        if max_steps is not None and waves >= max_steps:
            break
        waves += 1
        level_size = len(q)
        for _ in range(level_size):
            u = int(q.popleft())
            for v in sorted(graph.neighbors(u)):
                vi = int(v)
                if vi in blocked or vi in infected:
                    continue
                prob = _edge_activation_prob(u, vi, p, edge_probs, propagation_multiplier)
                if prob < 0 or prob > 1:
                    raise ValueError(f"Invalid activation probability {prob} on edge ({u},{vi})")
                if rng.random() < prob:
                    infected.add(vi)
                    q.append(vi)
    return infected, waves


def _community_infection_rates(
    infected: Set[int],
    graph: nx.Graph,
    communities: Mapping[int, int],
) -> Dict[int, float]:
    """Fraction of nodes in each structural community that are infected."""
    by_comm: Dict[int, List[int]] = {}
    for n in graph.nodes:
        nid = int(n)
        cid = int(communities.get(nid, -1))
        by_comm.setdefault(cid, []).append(nid)
    rates: Dict[int, float] = {}
    for cid, members in by_comm.items():
        if not members:
            continue
        rates[cid] = sum(1 for x in members if x in infected) / float(len(members))
    return rates


def simulate_ic(
    graph: nx.Graph,
    seeds: Sequence[int],
    *,
    blocked_nodes: Optional[Iterable[int]] = None,
    p: float = 0.1,
    n_runs: int = 64,
    communities: Optional[Mapping[int, int]] = None,
    random_seed: Optional[int] = None,
    max_steps: Optional[int] = None,
    edge_probs: Optional[nx.Graph] = None,
    scenario_harm_weight: float = 1.0,
    node_harm_weights: Optional[Mapping[int, float]] = None,
    propagation_multiplier: float = 1.0,
) -> ICResult:
    """
    Monte Carlo Independent Cascade on an undirected ``networkx.Graph``.

    Each wave, every node that became active in the previous wave attempts once
    per susceptible neighbor. Blocked nodes never infect or become infected.
    Seeds that are blocked are removed from the initial active set.

    Parameters
    ----------
    graph :
        Undirected graph (int node ids recommended).
    seeds :
        Candidate seeds; entries not in ``graph`` are skipped.
    blocked_nodes :
        Nodes that cannot be infected or propagate.
    p :
        Scalar activation probability when ``edge_probs`` is ``None``.
    n_runs :
        Monte Carlo replicates.
    communities :
        Optional ``node_id -> community_id`` for **structural** group summaries.
    random_seed :
        Base seed; run ``i`` uses ``random_seed + i`` for independent streams.
    max_steps :
        Optional cap on the number of diffusion waves.
    edge_probs :
        Graph with same edges; edge data uses ``p`` or ``weight`` in ``[0,1]``.

    Returns
    -------
    dict
        Includes legacy infection-count outputs plus optional harm-aware metrics:
        ``mean_total_harm``, ``std_total_harm``, ``mean_harm_per_node``,
        ``community_mean_harm_rate`` (or ``None``),
        ``mean_worst_community_harm_rate`` (or ``None``).
    """
    if graph.number_of_nodes() == 0:
        raise ValueError("graph has no nodes")
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph must be an undirected networkx.Graph")
    if nx.is_directed(graph):
        raise TypeError("graph must be undirected (got DiGraph)")

    if n_runs < 1:
        raise ValueError("n_runs must be >= 1")
    if edge_probs is None and (p < 0 or p > 1 or math.isnan(p)):
        raise ValueError("p must be in [0, 1] when edge_probs is None")
    if propagation_multiplier < 0 or math.isnan(propagation_multiplier):
        raise ValueError("propagation_multiplier must be non-negative")
    if scenario_harm_weight < 0 or math.isnan(scenario_harm_weight):
        raise ValueError("scenario_harm_weight must be non-negative")

    blocked: Set[int] = set(int(x) for x in (blocked_nodes or []))
    seed_list = _normalize_nodes(graph, seeds)

    n_nodes = graph.number_of_nodes()
    base_seed = 0 if random_seed is None else int(random_seed)

    infected_counts: List[int] = []
    total_harm_values: List[float] = []
    step_counts: List[int] = []
    comm_accum: Optional[Dict[int, List[float]]] = None
    harm_comm_accum: Optional[Dict[int, List[float]]] = None
    worst_comm_per_run: List[float] = []
    worst_harm_comm_per_run: List[float] = []
    if communities is not None:
        comm_accum = {}
        harm_comm_accum = {}

    for run_idx in range(n_runs):
        run_rng = np.random.default_rng(base_seed + run_idx)
        infected, n_wav = _ic_single_run(
            graph,
            seed_list,
            blocked,
            p,
            run_rng,
            edge_probs,
            max_steps,
            propagation_multiplier,
        )
        infected_counts.append(len(infected))
        total_harm_values.append(
            total_harm(
                infected,
                scenario_harm_weight=scenario_harm_weight,
                node_harm_weights=node_harm_weights,
            )
        )
        step_counts.append(n_wav)

        if communities is not None and comm_accum is not None and harm_comm_accum is not None:
            rates = _community_infection_rates(infected, graph, communities)
            harm_rates = community_harm_rates(
                infected,
                graph,
                communities,
                scenario_harm_weight=scenario_harm_weight,
                node_harm_weights=node_harm_weights,
            )
            for cid, r in rates.items():
                comm_accum.setdefault(cid, []).append(r)
            for cid, r in harm_rates.items():
                harm_comm_accum.setdefault(cid, []).append(r)
            if rates:
                worst_comm_per_run.append(float(max(rates.values())))
            else:
                worst_comm_per_run.append(0.0)
            if harm_rates:
                worst_harm_comm_per_run.append(float(max(harm_rates.values())))
            else:
                worst_harm_comm_per_run.append(0.0)

    counts_arr = np.asarray(infected_counts, dtype=float)
    harm_arr = np.asarray(total_harm_values, dtype=float)
    mean_infected = float(counts_arr.mean())
    std_infected = float(counts_arr.std(ddof=1)) if n_runs > 1 else 0.0
    mean_total_harm = float(harm_arr.mean())
    std_total_harm = float(harm_arr.std(ddof=1)) if n_runs > 1 else 0.0
    mean_infection_rate = mean_infected / float(n_nodes)
    mean_harm_per_node = mean_total_harm / float(n_nodes)
    mean_steps = float(np.mean(step_counts)) if step_counts else 0.0

    community_mean: Optional[Dict[str, float]]
    mean_worst_comm: Optional[float]
    community_harm_mean: Optional[Dict[str, float]]
    mean_worst_harm_comm: Optional[float]
    if communities is None or comm_accum is None:
        community_mean = None
        mean_worst_comm = None
        community_harm_mean = None
        mean_worst_harm_comm = None
    else:
        community_mean = {
            str(cid): float(np.mean(vals)) for cid, vals in sorted(comm_accum.items())
        }
        mean_worst_comm = float(np.mean(worst_comm_per_run)) if worst_comm_per_run else 0.0
        community_harm_mean = {
            str(cid): float(np.mean(vals))
            for cid, vals in sorted((harm_comm_accum or {}).items())
        }
        mean_worst_harm_comm = (
            float(np.mean(worst_harm_comm_per_run)) if worst_harm_comm_per_run else 0.0
        )

    config: Dict[str, Any] = {
        "n_runs": n_runs,
        "p": p,
        "random_seed": base_seed,
        "max_steps": max_steps,
        "n_nodes": n_nodes,
        "n_seeds_requested": len(seeds),
        "n_seeds_used": len(seed_list),
        "n_blocked": len(blocked),
        "edge_probs": edge_probs is not None,
        "communities": communities is not None,
        "scenario_harm_weight": scenario_harm_weight,
        "propagation_multiplier": propagation_multiplier,
        "node_harm_weights": node_harm_weights is not None,
    }

    return {
        "mean_infected": mean_infected,
        "std_infected": std_infected,
        "mean_total_harm": mean_total_harm,
        "std_total_harm": std_total_harm,
        "mean_harm_per_node": mean_harm_per_node,
        "mean_infection_rate": mean_infection_rate,
        "infected_counts": infected_counts,
        "community_mean_infection_rate": community_mean,
        "mean_worst_community_infection_rate": mean_worst_comm,
        "community_mean_harm_rate": community_harm_mean,
        "mean_worst_community_harm_rate": mean_worst_harm_comm,
        "total_harm_values": total_harm_values,
        "mean_steps": mean_steps,
        "config": config,
    }


def assign_uniform_edge_probability(graph: nx.Graph, p: float) -> nx.Graph:
    """Copy ``graph`` with uniform ``p`` on each edge (for ``edge_probs``)."""
    h = nx.Graph()
    h.add_nodes_from(graph.nodes(data=True))
    for u, v in graph.edges():
        h.add_edge(int(u), int(v), p=float(p))
    return h
