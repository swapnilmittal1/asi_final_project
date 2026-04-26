"""Load processed ego bundles from ``selected_egos.json``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set

import networkx as nx
import pandas as pd

from src.config import DATA_PROCESSED, EGOS_PROCESSED_DIR


@dataclass
class EgoBundle:
    ego_id: int
    label: str
    graph: nx.Graph
    communities: Dict[int, int]
    community_source: str
    matrix_role: str = "primary"


def _load_legacy_dev() -> EgoBundle:
    edges = pd.read_csv(DATA_PROCESSED / "twitter_dev_edges.csv")
    g = nx.from_pandas_edgelist(edges, "u", "v")
    df = pd.read_csv(DATA_PROCESSED / "twitter_dev_node_communities.csv")
    comm = {int(r["node_id"]): int(r["primary_circle_id"]) for _, r in df.iterrows()}
    meta = json.loads((DATA_PROCESSED / "twitter_dev_metadata.json").read_text(encoding="utf-8"))
    src = str(meta.get("community_source") or ("detected" if meta.get("circles_fallback_needed") else "circles"))
    return EgoBundle(
        ego_id=int(meta["ego_id"]),
        label="dev",
        graph=g,
        communities=comm,
        community_source=src,
        matrix_role="debugging",
    )


def _load_ego_dir(ego_id: int, label: str, entry: Dict[str, Any]) -> EgoBundle:
    base = EGOS_PROCESSED_DIR / str(ego_id)
    meta_path = base / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}; run python -m src.data.export_selected_egos")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    edges = pd.read_csv(base / "edges.csv")
    g = nx.from_pandas_edgelist(edges, "u", "v")
    df = pd.read_csv(base / "node_communities.csv")
    comm = {int(r["node_id"]): int(r["primary_circle_id"]) for _, r in df.iterrows()}
    role = str(entry.get("matrix_role", "primary"))
    return EgoBundle(
        ego_id=int(ego_id),
        label=label,
        graph=g,
        communities=comm,
        community_source=str(meta.get("community_source", "unknown")),
        matrix_role=role,
    )


def load_ego_from_manifest_entry(entry: Dict[str, Any]) -> EgoBundle:
    if entry.get("legacy_dev_paths"):
        return _load_legacy_dev()
    return _load_ego_dir(int(entry["ego_id"]), str(entry.get("label", "")), entry)


def load_manifest() -> Dict[str, Any]:
    path = DATA_PROCESSED / "selected_egos.json"
    return json.loads(path.read_text(encoding="utf-8"))


def validate_seeds_and_block(seeds: List[int], blocked: Set[int], graph: nx.Graph) -> None:
    s = set(seeds)
    if len(s) != len(seeds):
        raise ValueError("duplicate seeds")
    if s & blocked:
        raise ValueError(f"blocked overlaps seeds: {s & blocked}")
    if not s.issubset(graph.nodes):
        raise ValueError("seeds not subset of V")
    if not blocked.issubset(graph.nodes):
        raise ValueError("blocked not subset of V")
