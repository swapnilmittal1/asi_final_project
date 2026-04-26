"""
Publication-style figures and summary tables (matplotlib only).

Run after ``run_full_matrix.py`` and ``run_fairness_robustness.py``.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, OUTPUTS, PROJECT_ROOT

logger = logging.getLogger(__name__)

FIG_DIR = OUTPUTS / "figures"
TBL_DIR = OUTPUTS / "tables"


def _load_full() -> pd.DataFrame:
    p = TBL_DIR / "full_experiment_results.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    # Back-compat: older CSVs lack per-run stderr / budget diagnostics
    if "mean_infected_stderr" not in df.columns and "std_infected" in df.columns and "n_runs_eval" in df.columns:
        n = df["n_runs_eval"].replace(0, np.nan)
        df["mean_infected_stderr"] = df["std_infected"] / np.sqrt(n)
    return df


def _load_fairness() -> pd.DataFrame:
    p = TBL_DIR / "fairness_robustness_results.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def build_table_dataset_summary() -> None:
    coaid = PROJECT_ROOT / "outputs" / "tables" / "coaid_summary.csv"
    rows = []
    if coaid.exists():
        s = pd.read_csv(coaid).iloc[0].to_dict()
        rows.append({"dataset": "coaid_articles", "metric": "n_total", "value": s.get("n_total", "")})
        rows.append({"dataset": "coaid_articles", "metric": "n_fake", "value": s.get("n_fake", "")})
        rows.append({"dataset": "coaid_articles", "metric": "n_real", "value": s.get("n_real", "")})
    else:
        rows.append({"dataset": "coaid", "metric": "note", "value": "coaid_summary.csv missing"})
    pd.DataFrame(rows).to_csv(TBL_DIR / "table_dataset_summary.csv", index=False)


def build_table_graph_summary(df: pd.DataFrame) -> None:
    sub = df.groupby(["ego_id", "ego_label", "matrix_role", "community_source"]).agg(
        n_nodes=("n_nodes", "first"),
        n_edges=("n_edges", "first"),
    ).reset_index()
    sub.to_csv(TBL_DIR / "table_graph_summary.csv", index=False)


def build_table_main_results(df: pd.DataFrame) -> None:
    """Primary egos, medium regime, high_degree seeds, budget 10."""
    sub = df[
        (df["matrix_role"] == "primary")
        & (df["propagation_regime"] == "medium")
        & (df["seed_strategy"] == "high_degree")
        & (df["budget_requested"] == 10)
    ]
    agg_kw: dict = {
        "mean_infected_mean": ("mean_infected", "mean"),
        "mean_infected_std_across_egos": ("mean_infected", "std"),
        "mean_infected_stderr_of_mean": (
            "mean_infected",
            lambda s: float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else 0.0,
        ),
        "worst_comm_mean": ("mean_worst_community_infection_rate", "mean"),
        "n_ego_cells": ("ego_id", "count"),
    }
    if "mean_infected_stderr" in sub.columns:
        agg_kw["avg_mc_stderr"] = ("mean_infected_stderr", "mean")
    g = sub.groupby("method", as_index=False).agg(**agg_kw)
    g.to_csv(TBL_DIR / "table_main_results.csv", index=False)


def build_table_budget_shortfall(df: pd.DataFrame) -> None:
    """
    Rows where requested budget exceeds what was placed (``budget_k``) or graph/pool caps apply.
    """
    blk = df[df["method"] != "no_intervention"].copy()
    blk["shortfall_vs_request"] = blk["budget_requested"] - blk["budget_k"]
    pool_limits = False
    if "pool_may_limit_greedy_family" in blk.columns:
        pool_limits = blk["pool_may_limit_greedy_family"].astype(str).str.lower().isin({"true", "1"})
    elif "greedy_pool_size" in blk.columns and "budget_sanitized" in blk.columns:
        pool_limits = blk["greedy_pool_size"] < blk["budget_sanitized"]
    graph_cap = (
        blk["budget_requested"] > blk["max_blockers_graph"]
        if "max_blockers_graph" in blk.columns
        else pd.Series(False, index=blk.index)
    )
    short = blk[(blk["shortfall_vs_request"] > 0) | graph_cap | pool_limits]
    gcols = ["ego_label", "method", "budget_requested"]
    agg_dict: dict = {
        "n_cells": ("ego_id", "count"),
        "mean_shortfall": ("shortfall_vs_request", "mean"),
    }
    if "max_blockers_graph" in short.columns:
        agg_dict["max_blockers_graph"] = ("max_blockers_graph", "first")
    if "greedy_pool_size" in short.columns:
        agg_dict["greedy_pool_size"] = ("greedy_pool_size", "first")
    if "budget_sanitized" in short.columns:
        agg_dict["budget_sanitized"] = ("budget_sanitized", "first")
    agg = short.groupby(gcols, as_index=False).agg(**agg_dict).sort_values(gcols)
    agg.to_csv(TBL_DIR / "table_budget_shortfall.csv", index=False)


def build_fairness_case_study_table(df: pd.DataFrame) -> None:
    """
    Compact fairness-focused comparison on two larger circle egos (structural groups only).
    """
    want = {
        "greedy_blocking",
        "fairness_aware_greedy_blocking",
        "top_degree_blocking",
        "pagerank_blocking",
    }
    sub = df[
        (df["ego_label"].isin(["moderate_large_circles", "large_circles"]))
        & (df["matrix_role"] == "primary")
        & (df["propagation_regime"] == "medium")
        & (df["seed_strategy"] == "high_degree")
        & (df["budget_requested"] == 10)
        & (df["method"].isin(want))
    ].copy()
    cols = [
        "ego_label",
        "method",
        "mean_infected",
        "std_infected",
        "mean_worst_community_infection_rate",
        "budget_k",
        "budget_requested",
    ]
    for c in ("mean_infected_stderr", "budget_sanitized", "greedy_pool_size"):
        if c in sub.columns:
            cols.append(c)
    sub = sub[[c for c in cols if c in sub.columns]]
    sub.to_csv(TBL_DIR / "table_fairness_case_study.csv", index=False)


def build_table_fairness_tradeoff(df: pd.DataFrame) -> None:
    sub = df[
        (df["matrix_role"] == "primary")
        & (df["propagation_regime"] == "medium")
        & (df["budget_requested"] == 10)
    ]
    sub = sub[["method", "mean_infected", "mean_worst_community_infection_rate", "ego_label"]].copy()
    sub.to_csv(TBL_DIR / "table_fairness_tradeoff.csv", index=False)


def build_table_robustness_summary(df: pd.DataFrame) -> None:
    sub = df[(df["matrix_role"] == "primary") & (df["seed_strategy"] == "high_degree") & (df["budget_requested"] == 10)]
    g = sub.groupby(["propagation_regime", "method"], as_index=False).agg(
        mean_infected=("mean_infected", "mean"),
        worst_comm=("mean_worst_community_infection_rate", "mean"),
    )
    g.to_csv(TBL_DIR / "table_robustness_summary.csv", index=False)


def figure_budget_vs_infected(df: pd.DataFrame) -> None:
    ego_label = "moderate_large_circles"
    sub = df[
        (df["ego_label"] == ego_label)
        & (df["seed_strategy"] == "high_degree")
        & (df["propagation_regime"] == "medium")
    ]
    if sub.empty:
        logger.warning("Skipping budget curve: no rows for %s", ego_label)
        return
    agg = sub.groupby(["budget_requested", "method"], as_index=False)["mean_infected"].mean()
    pivot = agg.pivot(index="budget_requested", columns="method", values="mean_infected")
    pivot = pivot.sort_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for col in pivot.columns:
        ax.plot(pivot.index, pivot[col], marker="o", label=col)
    ax.set_xlabel("budget (requested)")
    ax.set_ylabel("mean infected")
    ax.set_title(f"Budget vs spread ({ego_label}, high_degree, medium p)")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    path = FIG_DIR / f"figure_budget_vs_infected_{ego_label}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    pivot.to_csv(TBL_DIR / "agg_budget_vs_infected_moderate_large.csv")
    logger.info("Wrote %s", path)


def figure_method_bar_medium(df: pd.DataFrame) -> None:
    sub = df[
        (df["matrix_role"] == "primary")
        & (df["propagation_regime"] == "medium")
        & (df["seed_strategy"] == "high_degree")
        & (df["budget_requested"] == 10)
    ]
    g = (
        sub.groupby("method", as_index=False)
        .agg(
            mean_infected=("mean_infected", "mean"),
            stderr_across_egos=(
                "mean_infected",
                lambda s: float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else 0.0,
            ),
        )
        .sort_values("mean_infected")
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y_pos = np.arange(len(g))
    ax.barh(y_pos, g["mean_infected"], xerr=g["stderr_across_egos"], capsize=3, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(g["method"])
    ax.set_xlabel("mean infected (± SE across primary egos)")
    ax.set_title("Methods under medium IC, high-degree seeds, k=10 (requested)")
    fig.tight_layout()
    path = FIG_DIR / "figure_method_comparison_bar_medium.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    g.to_csv(TBL_DIR / "agg_method_bar_medium.csv", index=False)
    logger.info("Wrote %s", path)


def figure_fairness_case_study(df: pd.DataFrame) -> None:
    """Second contribution: fairness vs efficiency on two larger egos (k=10, medium p)."""
    want = [
        "greedy_blocking",
        "fairness_aware_greedy_blocking",
        "top_degree_blocking",
        "pagerank_blocking",
    ]
    sub = df[
        (df["ego_label"].isin(["moderate_large_circles", "large_circles"]))
        & (df["propagation_regime"] == "medium")
        & (df["seed_strategy"] == "high_degree")
        & (df["budget_requested"] == 10)
        & (df["method"].isin(want))
    ]
    if sub.empty:
        logger.warning("Skipping fairness case-study figure (no rows)")
        return
    ego_labels = ["moderate_large_circles", "large_circles"]
    ego_short = ["moderate–large circles", "large circles"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)
    x = np.arange(len(want))
    width = 0.36
    for ax, ylab, col in zip(
        axes,
        ["Mean infected", "Worst-community infection rate\n(structural groups, not demographics)"],
        ["mean_infected", "mean_worst_community_infection_rate"],
    ):
        for i, (ego, lab) in enumerate(zip(ego_labels, ego_short)):
            e = sub[sub["ego_label"] == ego].set_index("method")
            heights = [float(e.loc[m, col]) if m in e.index else np.nan for m in want]
            err = None
            if col == "mean_infected" and "mean_infected_stderr" in e.columns:
                err = [float(e.loc[m, "mean_infected_stderr"]) if m in e.index else 0.0 for m in want]
            offset = (i - 0.5) * width
            ax.bar(x + offset, heights, width, label=lab, yerr=err, capsize=2, alpha=0.88)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_blocking", "").replace("_", "\n") for m in want], fontsize=8)
        ax.set_ylabel(ylab)
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].set_title("Fairness case study: k=10 requested, medium p")
    fig.suptitle("Compare greedy, fairness-greedy, top-degree, PageRank", y=1.02, fontsize=10)
    fig.tight_layout()
    path = FIG_DIR / "figure_fairness_case_study.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", path)


def figure_fairness_tradeoff(df: pd.DataFrame, fair: pd.DataFrame) -> None:
    sub = df[
        (df["matrix_role"] == "primary")
        & (df["propagation_regime"] == "medium")
        & (df["budget_requested"] == 10)
    ]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for method in sorted(sub["method"].unique()):
        m = sub[sub["method"] == method]
        ax.scatter(
            m["mean_infected"],
            m["mean_worst_community_infection_rate"],
            label=method,
            alpha=0.7,
            s=40,
        )
    if not fair.empty:
        fr = fair[fair["method"] == "fairness_aware_greedy_blocking"]
        ax.scatter(
            fr["mean_infected"],
            fr["mean_worst_community_infection_rate"],
            c="black",
            marker="x",
            s=60,
            label="fairness λ sweep",
        )
    ax.set_xlabel("mean infected")
    ax.set_ylabel("mean worst-community infection rate")
    ax.set_title("Efficiency vs worst-group harm (structural communities)")
    ax.legend(fontsize=6, loc="best")
    fig.tight_layout()
    path = FIG_DIR / "figure_fairness_tradeoff.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", path)


def figure_robustness_p(df: pd.DataFrame) -> None:
    sub = df[(df["matrix_role"] == "primary") & (df["seed_strategy"] == "high_degree") & (df["budget_requested"] == 10)]
    methods = [
        "no_intervention",
        "random_blocking",
        "top_degree_blocking",
        "pagerank_blocking",
        "greedy_blocking",
        "fairness_aware_greedy_blocking",
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in methods:
        ms = sub[sub["method"] == m]
        if ms.empty:
            continue
        g = (
            ms.groupby("propagation_regime", as_index=False)
            .agg(
                mean_infected=("mean_infected", "mean"),
                se=("mean_infected", lambda s: float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else 0.0),
            )
        )
        order = ["low", "medium", "high"]
        g["propagation_regime"] = pd.Categorical(g["propagation_regime"], categories=order, ordered=True)
        g = g.sort_values("propagation_regime")
        xs = np.arange(len(g))
        ax.errorbar(xs, g["mean_infected"], yerr=g["se"], marker="o", label=m, capsize=2, linewidth=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["low p", "medium p", "high p"])
    ax.set_ylabel("mean infected (± SE across primary egos)")
    ax.set_title("Propagation regime sensitivity (high_degree, k=10)")
    ax.legend(fontsize=6, loc="best")
    fig.tight_layout()
    path = FIG_DIR / "figure_robustness_p_values.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", path)


def figure_runtime_methods(df: pd.DataFrame) -> None:
    sub = df[(df["matrix_role"] == "primary")]
    g = sub.groupby("method", as_index=False)["runtime_seconds"].mean().sort_values("runtime_seconds", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(g["method"], g["runtime_seconds"])
    ax.set_xlabel("mean runtime per evaluation (s)")
    ax.set_title("IC evaluation cost by method (primary egos, all matrix cells)")
    fig.tight_layout()
    path = FIG_DIR / "figure_runtime_methods.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    g.to_csv(TBL_DIR / "agg_runtime_by_method.csv", index=False)
    logger.info("Wrote %s", path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TBL_DIR.mkdir(parents=True, exist_ok=True)

    df = _load_full()
    try:
        fair = _load_fairness()
    except FileNotFoundError:
        fair = pd.DataFrame()

    build_table_dataset_summary()
    build_table_graph_summary(df)
    build_table_main_results(df)
    build_table_fairness_tradeoff(df)
    build_table_robustness_summary(df)
    build_table_budget_shortfall(df)
    build_fairness_case_study_table(df)

    figure_budget_vs_infected(df)
    figure_method_bar_medium(df)
    figure_fairness_case_study(df)
    figure_fairness_tradeoff(df, fair)
    figure_robustness_p(df)
    figure_runtime_methods(df)


if __name__ == "__main__":
    main()
