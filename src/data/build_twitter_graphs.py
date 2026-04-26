"""
Catalog SNAP ego-Twitter graphs, select a development ego, and export artifacts.

Edges are modeled as **undirected** for the IC pipeline (modeling simplification).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import networkx as nx
import pandas as pd

from src.config import DATA_PROCESSED, OUTPUTS_TABLES, TWITTER_RAW

logger = logging.getLogger(__name__)


@dataclass
class EgoSummary:
    ego_id: int
    n_nodes: int
    n_edges: int
    density: float
    n_components: int
    largest_component_size: int
    avg_degree: float
    max_degree: int
    circles_file_exists: bool
    circles_nonempty: bool
    n_circles: int
    nodes_covered_by_circles: int


def parse_edges_file(path: Path) -> nx.Graph:
    """Load whitespace-separated edge list into an undirected simple graph."""
    g = nx.Graph()
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            if u == v:
                continue
            g.add_edge(u, v)
    return g


def parse_circles_file(path: Path) -> Dict[int, List[int]]:
    """
    Parse SNAP ``.circles`` format: ``circle_id TAB user TAB user ...`` per line.

    Returns mapping ``circle_id -> member node ids``. Empty file -> empty dict.
    """
    if not path.exists() or path.stat().st_size == 0:
        return {}
    circles: Dict[int, List[int]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if not parts:
                continue
            cid = int(parts[0])
            members = [int(x) for x in parts[1:] if x.strip()]
            circles[cid] = members
    return circles


def circles_to_node_membership(circles: Dict[int, List[int]]) -> Dict[int, List[int]]:
    """Invert to ``node_id -> sorted list of circle ids``."""
    node_to: Dict[int, List[int]] = {}
    for cid, members in circles.items():
        for n in members:
            node_to.setdefault(n, []).append(cid)
    for n, cids in node_to.items():
        node_to[n] = sorted(set(cids))
    return node_to


def primary_circle_id(node_to_circles: Dict[int, List[int]], node: int) -> int:
    """Use minimum circle id as a deterministic primary label when circles overlap."""
    cids = node_to_circles.get(node, [])
    return min(cids) if cids else -1


def summarize_graph(g: nx.Graph) -> Tuple[int, int, float, int, int, float, int]:
    """Return n_nodes, n_edges, density, n_components, largest_cc, avg_degree, max_degree."""
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()
    dens = nx.density(g) if n_nodes > 1 else 0.0
    comps = list(nx.connected_components(g))
    n_comp = len(comps)
    largest = max((len(c) for c in comps), default=0)
    degs = [d for _, d in g.degree()]
    avg_d = float(sum(degs) / n_nodes) if n_nodes else 0.0
    max_d = max(degs, default=0)
    return n_nodes, n_edges, dens, n_comp, largest, avg_d, max_d


def build_catalog(edge_paths: Iterable[Path]) -> pd.DataFrame:
    rows: List[dict] = []
    for ep in sorted(edge_paths):
        ego_id = int(ep.stem)
        circles_path = ep.with_suffix(".circles")
        circles_exist = circles_path.exists()
        nonempty = circles_exist and circles_path.stat().st_size > 0
        circ = parse_circles_file(circles_path) if circles_exist else {}
        n_circles = len(circ)
        covered = len(circles_to_node_membership(circ)) if circ else 0

        g = parse_edges_file(ep)
        n_nodes, n_edges, dens, n_comp, largest, avg_d, max_d = summarize_graph(g)
        rows.append(
            asdict(
                EgoSummary(
                    ego_id=ego_id,
                    n_nodes=n_nodes,
                    n_edges=n_edges,
                    density=dens,
                    n_components=n_comp,
                    largest_component_size=largest,
                    avg_degree=avg_d,
                    max_degree=max_d,
                    circles_file_exists=circles_exist,
                    circles_nonempty=nonempty,
                    n_circles=n_circles,
                    nodes_covered_by_circles=covered,
                )
            )
        )
    return pd.DataFrame(rows)


def choose_dev_ego(df: pd.DataFrame) -> int:
    """
    Prefer a **medium-sized** ego with **non-empty circles**.

    Chooses the ego whose ``n_edges`` is closest to the global median among rows
    with ``circles_nonempty`` and ``500 <= n_edges <= 4000`` (fast iteration band).
    """
    med = float(df["n_edges"].median())
    band = df[(df["n_edges"] >= 500) & (df["n_edges"] <= 4000) & (df["circles_nonempty"])]
    if band.empty:
        logger.warning(
            "No ego in [500,4000] edges with non-empty circles; relaxing to any nonempty circles."
        )
        band = df[df["circles_nonempty"]]
    if band.empty:
        logger.warning("All circles empty; choosing closest-to-median edge count overall.")
        band = df
    idx = (band["n_edges"] - med).abs().idxmin()
    ego_id = int(band.loc[idx, "ego_id"])
    logger.info("Selected development ego_id=%s (median n_edges=%s)", ego_id, med)
    return ego_id


def export_dev_graph(ego_id: int) -> None:
    """Write edgelist, community mapping, and metadata for the development ego."""
    edges_path = TWITTER_RAW / f"{ego_id}.edges"
    circles_path = TWITTER_RAW / f"{ego_id}.circles"
    g = parse_edges_file(edges_path)
    circles = parse_circles_file(circles_path)
    node_to_circles = circles_to_node_membership(circles)

    edge_rows = [(u, v) for u, v in g.edges()]
    pd.DataFrame(edge_rows, columns=["u", "v"]).to_csv(
        DATA_PROCESSED / "twitter_dev_edges.csv", index=False
    )

    primary_rows = []
    for n in sorted(g.nodes()):
        primary_rows.append(
            {
                "node_id": int(n),
                "primary_circle_id": primary_circle_id(node_to_circles, n),
                "circle_ids": json.dumps(node_to_circles.get(n, [])),
            }
        )
    pd.DataFrame(primary_rows).to_csv(
        DATA_PROCESSED / "twitter_dev_node_communities.csv", index=False
    )

    meta = {
        "ego_id": ego_id,
        "undirected": True,
        "modeling_note": (
            "SNAP ego edges are treated as undirected for IC; this is a modeling "
            "simplification, not a claim about Twitter's directed activity."
        ),
        "n_nodes": g.number_of_nodes(),
        "n_edges": g.number_of_edges(),
        "circles_nonempty": bool(circles),
        "n_circles": len(circles),
        "nodes_with_circle_label": sum(
            1 for n in g.nodes() if primary_circle_id(node_to_circles, n) >= 0
        ),
        "circles_fallback_needed": not bool(circles),
    }
    with (DATA_PROCESSED / "twitter_dev_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    OUTPUTS_TABLES.mkdir(parents=True, exist_ok=True)

    edge_paths = sorted(TWITTER_RAW.glob("*.edges"))
    if not edge_paths:
        raise FileNotFoundError(f"No .edges under {TWITTER_RAW}")

    catalog = build_catalog(edge_paths)
    catalog_path = OUTPUTS_TABLES / "twitter_ego_summary.csv"
    catalog.to_csv(catalog_path, index=False)
    logger.info("Wrote catalog %s (%d egos)", catalog_path, len(catalog))

    ego_id = choose_dev_ego(catalog)
    with (DATA_PROCESSED / "twitter_dev_ego_id.txt").open("w", encoding="utf-8") as f:
        f.write(str(ego_id))

    export_dev_graph(ego_id)


if __name__ == "__main__":
    main()
