"""Reproducible misinformation seed placement from scenario JSON fields."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, List

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


def build_seeds(
    graph: nx.Graph,
    strategy: str,
    n_seeds: int,
    communities: Dict[int, int],
    rng: np.random.Generator,
) -> List[int]:
    """Construct seed node list per scenario ``seed_strategy``."""
    nodes = [int(n) for n in graph.nodes]
    n_seeds = min(n_seeds, len(nodes))
    if n_seeds <= 0:
        return []

    if strategy == "random":
        pick = rng.choice(len(nodes), size=n_seeds, replace=False)
        return sorted(int(nodes[i]) for i in pick)

    if strategy == "high_degree":
        scored = [(n, int(graph.degree(n))) for n in nodes]
        scored.sort(key=lambda t: (-t[1], t[0]))
        return [scored[i][0] for i in range(n_seeds)]

    if strategy == "community_concentrated":
        pos = [n for n in nodes if communities.get(n, -1) >= 0]
        if not pos:
            raise RuntimeError(
                "community_concentrated seeding requires at least one node with community id >= 0; "
                "use detected communities or another scenario."
            )
        by_c = Counter(communities[n] for n in pos)
        best_c, _ = max(by_c.items(), key=lambda kv: kv[1])
        members = [n for n in pos if communities[n] == best_c]
        scored = [(n, int(graph.degree(n))) for n in members]
        scored.sort(key=lambda t: (-t[1], t[0]))
        chosen = [scored[i][0] for i in range(min(n_seeds, len(scored)))]
        if len(chosen) < n_seeds:
            rest = [n for n in nodes if n not in chosen]
            rest.sort(key=lambda n: (-int(graph.degree(n)), n))
            for n in rest:
                if len(chosen) >= n_seeds:
                    break
                chosen.append(n)
            logger.warning(
                "community_concentrated: padded seeds from global high-degree nodes to reach n_seeds=%d",
                n_seeds,
            )
        return sorted(chosen[:n_seeds])

    raise ValueError(f"Unknown seed_strategy: {strategy!r}")
