"""
Structural community detection fallback when SNAP ``.circles`` is empty.

Outputs are **network positions only**, not demographic groups.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Set

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

logger = logging.getLogger(__name__)


def greedy_modularity_partition(graph: nx.Graph) -> Dict[int, int]:
    """
    Assign each node an integer community id using greedy modularity maximization.

    Uses ``greedy_modularity_communities`` from NetworkX.
    """
    if graph.number_of_nodes() == 0:
        return {}
    if not isinstance(graph, nx.Graph) or nx.is_directed(graph):
        raise TypeError("graph must be an undirected networkx.Graph")
    comps = list(greedy_modularity_communities(graph))
    mapping: Dict[int, int] = {}
    for i, comm in enumerate(comps):
        for n in comm:
            mapping[int(n)] = i
    missing = set(int(x) for x in graph.nodes) - set(mapping.keys())
    if missing:
        max_id = max(mapping.values(), default=-1)
        for j, n in enumerate(sorted(missing)):
            mapping[n] = max_id + 1 + j
        logger.warning("Greedy modularity missed %d nodes; assigned singleton ids.", len(missing))
    return mapping
