"""
Final-report-ready artifacts from the targeted harm story sweep.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, Iterable, List, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import OUTPUTS

logger = logging.getLogger(__name__)

FIG_DIR = OUTPUTS / "figures"
TBL_DIR = OUTPUTS / "tables"

PRIMARY_SLICE: Dict[str, object] = {
    "ego_label": "large_circles",
    "seed_strategy": "community_concentrated",
    "propagation_regime": "high",
    "propagation_mode": "constant",
    "severity_regime": "high",
    "budget_requested": 10,
}

ROBUSTNESS_SLICE: Dict[str, object] = {
    "ego_label": "large_circles",
    "severity_regime": "high",
}

METHOD_LABELS: Dict[str, str] = {
    "top_degree_blocking": "Top-degree",
    "greedy_blocking": "Greedy",
    "fairness_aware_greedy_blocking": "Fair greedy",
    "harm_aware_greedy_blocking": "Harm greedy",
    "harm_aware_resilience_greedy_blocking": "Harm+resilience",
    "pagerank_blocking": "PageRank",
    "random_blocking": "Random",
    "no_intervention": "No intervention",
}


def _load_csv(name: str) -> pd.DataFrame:
    path = TBL_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _apply_slice(df: pd.DataFrame, selectors: Dict[str, object]) -> pd.DataFrame:
    sub = df.copy()
    for col, value in selectors.items():
        sub = sub[sub[col] == value]
    return sub


def _blocked_nodes(row: pd.Series) -> Set[int]:
    return set(int(x) for x in json.loads(str(row["blocked_nodes_json"])))


def _jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 1.0
    return len(sa & sb) / float(len(union))


def _method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_blocking", "").replace("_", " "))


def build_story_best_wins_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for keys, g in df.groupby(
        [
            "ego_label",
            "seed_strategy",
            "propagation_regime",
            "propagation_mode",
            "severity_regime",
            "budget_requested",
        ]
    ):
        g = g.set_index("method")
        if "greedy_blocking" not in g.index or "fairness_aware_greedy_blocking" not in g.index:
            continue
        for method in ["harm_aware_greedy_blocking", "harm_aware_resilience_greedy_blocking"]:
            if method not in g.index:
                continue
            rows.append(
                {
                    "ego_label": keys[0],
                    "seed_strategy": keys[1],
                    "propagation_regime": keys[2],
                    "propagation_mode": keys[3],
                    "severity_regime": keys[4],
                    "budget_requested": keys[5],
                    "method": method,
                    "mean_total_harm": float(g.loc[method, "mean_total_harm"]),
                    "mean_worst_community_harm_rate": float(
                        g.loc[method, "mean_worst_community_harm_rate"]
                    ),
                    "delta_harm_vs_greedy": float(
                        g.loc[method, "mean_total_harm"] - g.loc["greedy_blocking", "mean_total_harm"]
                    ),
                    "delta_worst_harm_vs_greedy": float(
                        g.loc[method, "mean_worst_community_harm_rate"]
                        - g.loc["greedy_blocking", "mean_worst_community_harm_rate"]
                    ),
                    "delta_harm_vs_fairness": float(
                        g.loc[method, "mean_total_harm"]
                        - g.loc["fairness_aware_greedy_blocking", "mean_total_harm"]
                    ),
                    "delta_worst_harm_vs_fairness": float(
                        g.loc[method, "mean_worst_community_harm_rate"]
                        - g.loc["fairness_aware_greedy_blocking", "mean_worst_community_harm_rate"]
                    ),
                }
            )
    out = pd.DataFrame(rows).sort_values(
        ["delta_harm_vs_greedy", "delta_worst_harm_vs_greedy"]
    )
    out.to_csv(TBL_DIR / "table_harm_story_best_wins.csv", index=False)
    return out


def build_story_budget_table(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.assign(shortfall_vs_request=df["budget_requested"] - df["budget_k"])
        .groupby(
            ["ego_label", "seed_strategy", "severity_regime", "budget_requested", "method"],
            as_index=False,
        )
        .agg(
            mean_shortfall=("shortfall_vs_request", "mean"),
            greedy_pool_size=("greedy_pool_size", "mean"),
            budget_sanitized=("budget_sanitized", "mean"),
        )
        .sort_values(["ego_label", "seed_strategy", "severity_regime", "budget_requested", "method"])
    )
    out.to_csv(TBL_DIR / "table_harm_story_budget_transparency.csv", index=False)
    return out


def build_top_degree_competitiveness_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify slices where top-degree is best or nearly best on total harm.
    """
    rows: List[dict] = []
    for keys, g in df.groupby(
        [
            "ego_label",
            "seed_strategy",
            "propagation_regime",
            "propagation_mode",
            "severity_regime",
            "budget_requested",
        ]
    ):
        g = g.set_index("method")
        if "top_degree_blocking" not in g.index:
            continue
        best_method = str(g["mean_total_harm"].idxmin())
        best_harm = float(g["mean_total_harm"].min())
        top_degree_harm = float(g.loc["top_degree_blocking", "mean_total_harm"])
        rows.append(
            {
                "ego_label": keys[0],
                "seed_strategy": keys[1],
                "propagation_regime": keys[2],
                "propagation_mode": keys[3],
                "severity_regime": keys[4],
                "budget_requested": keys[5],
                "best_method": best_method,
                "top_degree_harm": top_degree_harm,
                "best_harm": best_harm,
                "top_degree_gap": top_degree_harm - best_harm,
                "top_degree_is_best": best_method == "top_degree_blocking",
            }
        )
    out = pd.DataFrame(rows).sort_values(
        ["top_degree_gap", "ego_label", "budget_requested", "severity_regime"]
    )
    out.to_csv(TBL_DIR / "table_harm_story_top_degree_competitiveness.csv", index=False)
    return out


def build_primary_slice_table(df: pd.DataFrame) -> pd.DataFrame:
    sub = _apply_slice(df, PRIMARY_SLICE).copy()
    sub = sub.sort_values("mean_total_harm")
    sub.to_csv(TBL_DIR / "table_harm_story_primary_slice.csv", index=False)
    return sub


def build_primary_claim_sheet(sub: pd.DataFrame) -> pd.DataFrame:
    """
    Small report-ready comparison sheet for the hero slice.
    """
    g = sub.set_index("method")
    greedy = g.loc["greedy_blocking"]
    fairness = g.loc["fairness_aware_greedy_blocking"]
    top_degree = g.loc["top_degree_blocking"]
    harm = g.loc["harm_aware_greedy_blocking"]
    harm_res = g.loc["harm_aware_resilience_greedy_blocking"]

    rows = [
        {
            "comparison": "harm_aware_greedy_vs_greedy",
            "delta_total_harm": float(harm["mean_total_harm"] - greedy["mean_total_harm"]),
            "delta_worst_community_harm": float(
                harm["mean_worst_community_harm_rate"] - greedy["mean_worst_community_harm_rate"]
            ),
        },
        {
            "comparison": "harm_aware_greedy_vs_fairness_greedy",
            "delta_total_harm": float(harm["mean_total_harm"] - fairness["mean_total_harm"]),
            "delta_worst_community_harm": float(
                harm["mean_worst_community_harm_rate"] - fairness["mean_worst_community_harm_rate"]
            ),
        },
        {
            "comparison": "harm_aware_greedy_vs_top_degree",
            "delta_total_harm": float(harm["mean_total_harm"] - top_degree["mean_total_harm"]),
            "delta_worst_community_harm": float(
                harm["mean_worst_community_harm_rate"] - top_degree["mean_worst_community_harm_rate"]
            ),
        },
        {
            "comparison": "harm_resilience_vs_greedy",
            "delta_total_harm": float(harm_res["mean_total_harm"] - greedy["mean_total_harm"]),
            "delta_worst_community_harm": float(
                harm_res["mean_worst_community_harm_rate"] - greedy["mean_worst_community_harm_rate"]
            ),
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(TBL_DIR / "table_harm_story_claim_sheet.csv", index=False)
    return out


def build_primary_overlap_table(sub: pd.DataFrame) -> pd.DataFrame:
    by_method = {str(r["method"]): r for _, r in sub.iterrows()}
    greedy_nodes = _blocked_nodes(pd.Series(by_method["greedy_blocking"]))
    rows: List[dict] = []
    for method, row in by_method.items():
        nodes = _blocked_nodes(pd.Series(row))
        rows.append(
            {
                "method": method,
                "budget_k": int(row["budget_k"]),
                "jaccard_with_greedy": _jaccard(nodes, greedy_nodes),
                "overlap_count_with_greedy": len(nodes & greedy_nodes),
                "unique_vs_greedy_json": json.dumps(sorted(nodes - greedy_nodes)),
                "blocked_nodes_json": row["blocked_nodes_json"],
            }
        )
    out = pd.DataFrame(rows).sort_values("jaccard_with_greedy", ascending=False)
    out.to_csv(TBL_DIR / "table_harm_story_blocked_set_changes.csv", index=False)
    return out


def figure_primary_comparison(sub: pd.DataFrame) -> None:
    methods = [
        "top_degree_blocking",
        "greedy_blocking",
        "fairness_aware_greedy_blocking",
        "harm_aware_greedy_blocking",
        "harm_aware_resilience_greedy_blocking",
    ]
    plot = sub[sub["method"].isin(methods)].copy()
    plot = plot.set_index("method").loc[methods].reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    x = np.arange(len(plot))
    axes[0].bar(
        x,
        plot["mean_total_harm"],
        yerr=plot["mean_total_harm_stderr"],
        capsize=3,
        alpha=0.88,
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([_method_label(m).replace(" ", "\n") for m in plot["method"]], fontsize=8)
    axes[0].set_ylabel("Mean total harm\n(lower is better)")
    axes[0].set_title("Total harm")

    axes[1].bar(x, plot["mean_worst_community_harm_rate"], alpha=0.88)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([_method_label(m).replace(" ", "\n") for m in plot["method"]], fontsize=8)
    axes[1].set_ylabel("Worst-community harm rate\n(lower is better)")
    axes[1].set_title("Concentrated harm")

    fig.suptitle("Primary comparison", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_harm_story_primary_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_primary_overlap(overlap_df: pd.DataFrame) -> None:
    plot = overlap_df[
        overlap_df["method"].isin(
            [
                "top_degree_blocking",
                "fairness_aware_greedy_blocking",
                "harm_aware_greedy_blocking",
                "harm_aware_resilience_greedy_blocking",
            ]
        )
    ].copy()
    fig, ax = plt.subplots(figsize=(7, 4.1))
    x = np.arange(len(plot))
    ax.bar(x, plot["jaccard_with_greedy"], alpha=0.88)
    ax.set_ylabel("Blocked-set overlap with\nbinary greedy (Jaccard)")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([_method_label(m).replace(" ", "\n") for m in plot["method"]], fontsize=8)
    ax.set_title("Changed blocked sets")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_harm_story_blocked_set_overlap.png", dpi=150)
    plt.close(fig)


def figure_story_tradeoff(df: pd.DataFrame) -> None:
    methods = [
        "greedy_blocking",
        "fairness_aware_greedy_blocking",
        "harm_aware_greedy_blocking",
        "harm_aware_resilience_greedy_blocking",
        "top_degree_blocking",
    ]
    sub = df[
        (df["budget_requested"] == 10)
        & (df["severity_regime"] == "high")
        & (df["method"].isin(methods))
    ]
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for method in methods:
        m = sub[sub["method"] == method]
        ax.scatter(
            m["mean_total_harm"],
            m["mean_worst_community_harm_rate"],
            label=_method_label(method),
            alpha=0.8,
            s=42,
        )
    ax.set_xlabel("Mean total harm (lower is better)")
    ax.set_ylabel("Worst-community harm rate (lower is better)")
    ax.set_title("Total harm vs concentrated harm")
    ax.legend(fontsize=7, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_harm_story_tradeoff.png", dpi=150)
    plt.close(fig)


def figure_story_propagation_robustness(df: pd.DataFrame) -> None:
    sub = _apply_slice(df, ROBUSTNESS_SLICE).copy()
    methods = [
        "greedy_blocking",
        "fairness_aware_greedy_blocking",
        "harm_aware_greedy_blocking",
        "harm_aware_resilience_greedy_blocking",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2), sharex=True)
    order = ["constant", "bucket", "linear"]
    for method in methods:
        m = (
            sub[sub["method"] == method]
            .groupby("propagation_mode", as_index=False)
            .agg(
                mean_total_harm=("mean_total_harm", "mean"),
                mean_worst_community_harm_rate=("mean_worst_community_harm_rate", "mean"),
            )
        )
        m["propagation_mode"] = pd.Categorical(m["propagation_mode"], categories=order, ordered=True)
        m = m.sort_values("propagation_mode")
        x = np.arange(len(m))
        axes[0].plot(x, m["mean_total_harm"], marker="o", label=_method_label(method))
        axes[1].plot(x, m["mean_worst_community_harm_rate"], marker="o", label=_method_label(method))
    for ax, ylabel in zip(
        axes,
        ["Mean total harm\n(lower is better)", "Worst-community harm rate\n(lower is better)"],
    ):
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order)
        ax.set_xlabel("Propagation mode")
        ax.set_ylabel(ylabel)
    axes[0].set_title("Total harm")
    axes[1].set_title("Concentrated harm")
    axes[1].legend(fontsize=7, loc="best", ncol=2)
    fig.suptitle("Propagation robustness", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_harm_story_propagation_robustness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TBL_DIR.mkdir(parents=True, exist_ok=True)

    story_df = _load_csv("harm_story_sweep_results.csv")
    sensitivity_df = _load_csv("severity_sensitivity_results.csv")

    build_story_best_wins_table(story_df)
    build_story_budget_table(story_df)
    build_top_degree_competitiveness_table(story_df)
    primary = build_primary_slice_table(story_df)
    build_primary_claim_sheet(primary)
    overlap = build_primary_overlap_table(primary)

    figure_primary_comparison(primary)
    figure_primary_overlap(overlap)
    figure_story_tradeoff(story_df)
    figure_story_propagation_robustness(sensitivity_df)
    logger.info("Wrote harm story figures and tables")


if __name__ == "__main__":
    main()
