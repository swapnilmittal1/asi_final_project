"""
Helpers for severity-weighted harm summaries on infected node sets.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Set

import networkx as nx


def node_harm_weight(
    node_id: int,
    *,
    scenario_harm_weight: float,
    node_harm_weights: Optional[Mapping[int, float]] = None,
) -> float:
    """
    Harm contribution for a single infected node.

    ``scenario_harm_weight`` is the main severity knob; optional node-specific
    weights allow future extensions without changing the public API again.
    """
    node_weight = 1.0
    if node_harm_weights is not None:
        node_weight = float(node_harm_weights.get(int(node_id), 1.0))
    return float(scenario_harm_weight) * node_weight


def total_harm(
    infected: Set[int],
    *,
    scenario_harm_weight: float,
    node_harm_weights: Optional[Mapping[int, float]] = None,
) -> float:
    """Total severity-weighted harm over infected nodes."""
    return sum(
        node_harm_weight(
            int(node),
            scenario_harm_weight=scenario_harm_weight,
            node_harm_weights=node_harm_weights,
        )
        for node in infected
    )


def community_harm_rates(
    infected: Set[int],
    graph: nx.Graph,
    communities: Mapping[int, int],
    *,
    scenario_harm_weight: float,
    node_harm_weights: Optional[Mapping[int, float]] = None,
) -> Dict[int, float]:
    """
    Mean harm per node inside each structural community.

    This normalizes by community size so larger communities do not dominate
    merely because they contain more nodes.
    """
    members_by_comm: Dict[int, list[int]] = {}
    for node in graph.nodes:
        node_id = int(node)
        members_by_comm.setdefault(int(communities.get(node_id, -1)), []).append(node_id)

    harm_rates: Dict[int, float] = {}
    for comm_id, members in members_by_comm.items():
        if not members:
            continue
        total = sum(
            node_harm_weight(
                member,
                scenario_harm_weight=scenario_harm_weight,
                node_harm_weights=node_harm_weights,
            )
            for member in members
            if member in infected
        )
        harm_rates[comm_id] = total / float(len(members))
    return harm_rates
