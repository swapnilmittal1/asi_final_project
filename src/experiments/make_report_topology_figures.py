"""
Report topology figures for the final harm-aware case study.

Run after ``run_harm_story_sweep.py`` and ``make_harm_story_figures.py`` so
``table_harm_story_primary_slice.csv`` is available.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Set

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.config import EGOS_PROCESSED_DIR, OUTPUTS

logger = logging.getLogger(__name__)

FIG_DIR = OUTPUTS / "figures"
TBL_DIR = OUTPUTS / "tables"
PRIMARY_EGO_ID = 88639412


def _load_graph_and_communities(ego_id: int = PRIMARY_EGO_ID) -> tuple[nx.Graph, Dict[int, int]]:
    ego_dir = EGOS_PROCESSED_DIR / str(ego_id)
    edges = pd.read_csv(ego_dir / "edges.csv")
    communities = pd.read_csv(ego_dir / "node_communities.csv")
    graph = nx.from_pandas_edgelist(edges, "u", "v")
    comm_map = dict(
        zip(
            communities["node_id"].astype(int),
            communities["primary_circle_id"].astype(int),
        )
    )
    return graph, comm_map


def _node_set(row: pd.Series, column: str) -> Set[int]:
    return set(int(x) for x in json.loads(str(row[column])))


def _top_communities(comm_map: Dict[int, int], n: int = 8) -> list[int]:
    counts = pd.Series(comm_map).value_counts()
    return [int(c) for c in counts.index.tolist() if int(c) >= 0][:n]


def _draw_base_graph(graph: nx.Graph, pos: Dict[int, np.ndarray]) -> None:
    nx.draw_networkx_edges(graph, pos, width=0.35, alpha=0.16, edge_color="#777777")


def build_topology_communities() -> None:
    graph, comm_map = _load_graph_and_communities()
    pos = nx.spring_layout(graph, seed=42, k=0.24, iterations=300)
    top_comms = _top_communities(comm_map)
    palette = plt.get_cmap("tab10")
    node_colors = []
    for node in graph.nodes:
        cid = int(comm_map.get(int(node), -1))
        if cid in top_comms:
            node_colors.append(palette(top_comms.index(cid) % 10))
        elif cid == -1:
            node_colors.append((0.78, 0.78, 0.78, 1.0))
        else:
            node_colors.append((0.55, 0.55, 0.55, 1.0))

    plt.figure(figsize=(9, 7))
    _draw_base_graph(graph, pos)
    nx.draw_networkx_nodes(graph, pos, node_size=38, node_color=node_colors, linewidths=0)
    handles = [Patch(facecolor=palette(i % 10), label=f"Community {cid}") for i, cid in enumerate(top_comms[:6])]
    handles.append(Patch(facecolor="#c7c7c7", label="Unlabeled/other"))
    plt.legend(handles=handles, loc="lower left", frameon=True, fontsize=8)
    plt.title("SNAP Ego-Twitter Topology: large_circles Structural Communities")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "report_topology_communities_large_circles.png", dpi=220)
    plt.close()


def build_seed_and_control_overlays() -> None:
    graph, comm_map = _load_graph_and_communities()
    pos = nx.spring_layout(graph, seed=42, k=0.24, iterations=300)
    primary = pd.read_csv(TBL_DIR / "table_harm_story_primary_slice.csv")
    harm = primary.loc[primary["method"] == "harm_aware_greedy_blocking"].iloc[0]
    greedy = primary.loc[primary["method"] == "greedy_blocking"].iloc[0]
    seeds = _node_set(harm, "seeds_json")
    harm_blocked = _node_set(harm, "blocked_nodes_json")
    greedy_blocked = _node_set(greedy, "blocked_nodes_json")

    plt.figure(figsize=(9, 7))
    _draw_base_graph(graph, pos)
    base_nodes = [n for n in graph.nodes if int(n) not in seeds and int(n) not in harm_blocked]
    nx.draw_networkx_nodes(graph, pos, nodelist=base_nodes, node_size=26, node_color="#d0d0d0", linewidths=0)
    nx.draw_networkx_nodes(graph, pos, nodelist=list(seeds), node_size=110, node_color="#d62728", edgecolors="white", linewidths=0.8)
    nx.draw_networkx_nodes(graph, pos, nodelist=list(harm_blocked), node_size=120, node_color="#1f77b4", edgecolors="black", linewidths=0.6, node_shape="s")
    plt.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", label="Other nodes", markerfacecolor="#d0d0d0", markersize=7),
            Line2D([0], [0], marker="o", color="w", label="Misinformation seeds", markerfacecolor="#d62728", markeredgecolor="white", markersize=9),
            Line2D([0], [0], marker="s", color="w", label="Harm-aware blocked nodes", markerfacecolor="#1f77b4", markeredgecolor="black", markersize=9),
        ],
        loc="lower left",
        frameon=True,
        fontsize=8,
    )
    plt.title("Community-Concentrated Seeds and Harm-Aware Controls")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "report_topology_seeds_harm_blockers.png", dpi=220)
    plt.close()

    common = harm_blocked & greedy_blocked
    harm_only = harm_blocked - greedy_blocked
    greedy_only = greedy_blocked - harm_blocked
    plt.figure(figsize=(9, 7))
    _draw_base_graph(graph, pos)
    highlighted = common | harm_only | greedy_only | seeds
    base_nodes = [n for n in graph.nodes if int(n) not in highlighted]
    nx.draw_networkx_nodes(graph, pos, nodelist=base_nodes, node_size=24, node_color="#d9d9d9", linewidths=0)
    nx.draw_networkx_nodes(graph, pos, nodelist=list(seeds), node_size=80, node_color="#d62728", edgecolors="white", linewidths=0.8)
    nx.draw_networkx_nodes(graph, pos, nodelist=list(common), node_size=115, node_color="#9467bd", edgecolors="black", linewidths=0.5, node_shape="s")
    nx.draw_networkx_nodes(graph, pos, nodelist=list(harm_only), node_size=140, node_color="#1f77b4", edgecolors="black", linewidths=0.7, node_shape="D")
    nx.draw_networkx_nodes(graph, pos, nodelist=list(greedy_only), node_size=140, node_color="#ff7f0e", edgecolors="black", linewidths=0.7, node_shape="^")
    plt.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", label="Seeds", markerfacecolor="#d62728", markeredgecolor="white", markersize=8),
            Line2D([0], [0], marker="s", color="w", label="Blocked by both", markerfacecolor="#9467bd", markeredgecolor="black", markersize=8),
            Line2D([0], [0], marker="D", color="w", label="Harm-aware only", markerfacecolor="#1f77b4", markeredgecolor="black", markersize=8),
            Line2D([0], [0], marker="^", color="w", label="Binary greedy only", markerfacecolor="#ff7f0e", markeredgecolor="black", markersize=8),
        ],
        loc="lower left",
        frameon=True,
        fontsize=8,
    )
    plt.title("Blocked-Set Difference: Binary Greedy vs Harm-Aware Greedy")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "report_topology_blocker_overlap.png", dpi=220)
    plt.close()

    seed_comms = [comm_map.get(int(s), -1) for s in seeds]
    seed_comm = int(pd.Series(seed_comms).mode().iloc[0])
    comm_size = pd.Series(comm_map).value_counts().sort_values(ascending=False).head(12)
    colors = ["#d62728" if int(c) == seed_comm else "#7f7f7f" for c in comm_size.index]
    plt.figure(figsize=(8, 4.8))
    plt.bar([str(c) for c in comm_size.index], comm_size.values, color=colors)
    plt.xlabel("Structural community id")
    plt.ylabel("Number of nodes")
    plt.title("Community-Concentrated Seeding Uses the Largest Structural Community")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "report_community_size_seed_concentration.png", dpi=220)
    plt.close()


def build_primary_harm_bars() -> None:
    primary = pd.read_csv(TBL_DIR / "table_harm_story_primary_slice.csv")
    order = [
        "no_intervention",
        "random_blocking",
        "pagerank_blocking",
        "fairness_aware_greedy_blocking",
        "greedy_blocking",
        "top_degree_blocking",
        "harm_aware_resilience_greedy_blocking",
        "harm_aware_greedy_blocking",
    ]
    primary["method"] = pd.Categorical(primary["method"], categories=order, ordered=True)
    primary = primary.sort_values("method")
    labels = [str(m).replace("_blocking", "").replace("_", "\n") for m in primary["method"]]
    x = np.arange(len(primary))
    fig, ax1 = plt.subplots(figsize=(10, 5.2))
    ax1.bar(x - 0.18, primary["mean_total_harm"], width=0.36, color="#4c78a8", label="Mean total harm")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, primary["mean_worst_community_harm_rate"], width=0.36, color="#f58518", label="Worst-community harm rate")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax1.set_ylabel("Mean total harm")
    ax2.set_ylabel("Worst-community harm rate")
    ax1.set_title("Primary High-Severity Slice: Total Harm and Concentrated Harm")
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "report_primary_slice_harm_bars.png", dpi=220)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    build_topology_communities()
    build_seed_and_control_overlays()
    build_primary_harm_bars()
    logger.info("Wrote report topology figures to %s", FIG_DIR)


if __name__ == "__main__":
    main()
