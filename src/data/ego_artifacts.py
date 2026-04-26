"""
Export SNAP ego networks to processed artifacts (edges + structural communities).

Community source is either SNAP ``circles`` (``primary_circle_id``) or greedy modularity
when circles are empty.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import pandas as pd

from src.config import DATA_PROCESSED, TWITTER_RAW
from src.data.community_detection import greedy_modularity_partition

logger = logging.getLogger(__name__)


def parse_edges_file(path: Path) -> nx.Graph:
    g = nx.Graph()
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            if u != v:
                g.add_edge(u, v)
    return g


def parse_circles_file(path: Path) -> Dict[int, List[int]]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    circles: Dict[int, List[int]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            cid = int(parts[0])
            members = [int(x) for x in parts[1:] if x.strip()]
            circles[cid] = members
    return circles


def circles_to_primary_map(graph: nx.Graph, circles: Dict[int, List[int]]) -> Dict[int, int]:
    node_to_circles: Dict[int, List[int]] = {}
    for cid, members in circles.items():
        for n in members:
            node_to_circles.setdefault(int(n), []).append(cid)
    for n in node_to_circles:
        node_to_circles[n] = sorted(set(node_to_circles[n]))
    primary: Dict[int, int] = {}
    for n in graph.nodes:
        nid = int(n)
        cids = node_to_circles.get(nid, [])
        primary[nid] = min(cids) if cids else -1
    return primary


def build_communities_for_ego(ego_id: int, graph: nx.Graph) -> Tuple[Dict[int, int], str]:
    circles_path = TWITTER_RAW / f"{ego_id}.circles"
    circles = parse_circles_file(circles_path)
    if circles:
        return circles_to_primary_map(graph, circles), "circles"
    part = greedy_modularity_partition(graph)
    return part, "detected"


def export_ego_artifacts(ego_id: int, out_dir: Path) -> None:
    """Write ``edges.csv``, ``node_communities.csv``, ``metadata.json`` under ``out_dir``."""
    edges_path = TWITTER_RAW / f"{ego_id}.edges"
    if not edges_path.exists():
        raise FileNotFoundError(edges_path)
    graph = parse_edges_file(edges_path)
    comm_map, source = build_communities_for_ego(ego_id, graph)

    out_dir.mkdir(parents=True, exist_ok=True)
    edge_rows = [(u, v) for u, v in graph.edges()]
    pd.DataFrame(edge_rows, columns=["u", "v"]).to_csv(out_dir / "edges.csv", index=False)

    rows = []
    for n in sorted(graph.nodes()):
        nid = int(n)
        cid = int(comm_map.get(nid, -1))
        rows.append({"node_id": nid, "primary_circle_id": cid, "circle_ids": json.dumps([cid])})
    pd.DataFrame(rows).to_csv(out_dir / "node_communities.csv", index=False)

    meta = {
        "ego_id": ego_id,
        "undirected": True,
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "community_source": source,
        "n_community_labels": len(set(comm_map.values())),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Exported ego %s to %s (source=%s)", ego_id, out_dir, source)
