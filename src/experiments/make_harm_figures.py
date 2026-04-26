"""
Figures and tables for the severity / harm extension.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import COAID_SEVERITY_REGIMES_CSV, OUTPUTS

logger = logging.getLogger(__name__)

FIG_DIR = OUTPUTS / "figures"
TBL_DIR = OUTPUTS / "tables"


def _load_csv(name: str) -> pd.DataFrame:
    path = TBL_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def build_table_binary_vs_harm_summary(df: pd.DataFrame) -> None:
    sub = df[(df["budget_requested"] == 10) & (df["severity_regime"] == "medium")]
    g = (
        sub.groupby("method", as_index=False)
        .agg(
            mean_infected=("mean_infected", "mean"),
            se_infected=("mean_infected", lambda s: float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else 0.0),
            mean_total_harm=("mean_total_harm", "mean"),
            se_total_harm=("mean_total_harm", lambda s: float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else 0.0),
            mean_worst_community_harm_rate=("mean_worst_community_harm_rate", "mean"),
            n_cells=("ego_id", "count"),
        )
        .sort_values("mean_total_harm")
    )
    g.to_csv(TBL_DIR / "table_binary_vs_harm_summary.csv", index=False)


def build_table_severity_regime_summary() -> None:
    df = pd.read_csv(COAID_SEVERITY_REGIMES_CSV)
    df.to_csv(TBL_DIR / "table_severity_regime_summary.csv", index=False)


def build_table_harm_budget_shortfall(df: pd.DataFrame) -> None:
    out = (
        df.assign(shortfall_vs_request=df["budget_requested"] - df["budget_k"])
        .groupby(["severity_regime", "method", "budget_requested"], as_index=False)
        .agg(
            n_cells=("ego_id", "count"),
            mean_shortfall=("shortfall_vs_request", "mean"),
            greedy_pool_size=("greedy_pool_size", "mean"),
            budget_sanitized=("budget_sanitized", "mean"),
        )
        .sort_values(["severity_regime", "budget_requested", "method"])
    )
    out.to_csv(TBL_DIR / "table_harm_budget_shortfall.csv", index=False)


def build_table_harm_case_study(df: pd.DataFrame) -> None:
    methods = [
        "top_degree_blocking",
        "pagerank_blocking",
        "greedy_blocking",
        "fairness_aware_greedy_blocking",
        "harm_aware_greedy_blocking",
        "harm_aware_resilience_greedy_blocking",
    ]
    sub = df[
        (df["ego_label"].isin(["moderate_large_circles", "large_circles"]))
        & (df["budget_requested"] == 10)
        & (df["severity_regime"] == "high")
        & (df["method"].isin(methods))
    ].copy()
    sub.to_csv(TBL_DIR / "table_harm_case_study.csv", index=False)


def build_table_harm_robustness(df: pd.DataFrame) -> None:
    g = (
        df.groupby(["propagation_mode", "severity_regime", "method"], as_index=False)
        .agg(
            mean_total_harm=("mean_total_harm", "mean"),
            mean_infected=("mean_infected", "mean"),
            mean_worst_community_harm_rate=("mean_worst_community_harm_rate", "mean"),
        )
        .sort_values(["method", "propagation_mode", "severity_regime"])
    )
    g.to_csv(TBL_DIR / "table_harm_robustness.csv", index=False)


def figure_budget_vs_infected(df: pd.DataFrame) -> None:
    sub = df[
        (df["ego_label"] == "moderate_large_circles")
        & (df["severity_regime"] == "high")
        & (
            df["method"].isin(
                [
                    "top_degree_blocking",
                    "greedy_blocking",
                    "harm_aware_greedy_blocking",
                    "harm_aware_resilience_greedy_blocking",
                ]
            )
        )
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in sorted(sub["method"].unique()):
        m = sub[sub["method"] == method].sort_values("budget_requested")
        ax.plot(m["budget_requested"], m["mean_infected"], marker="o", label=method)
    ax.set_xlabel("Requested blockers")
    ax.set_ylabel("Mean infected")
    ax.set_title("Budget vs infected count (moderate_large_circles, high severity)")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_budget_vs_infected_harm.png", dpi=150)
    plt.close(fig)


def figure_budget_vs_total_harm(df: pd.DataFrame) -> None:
    sub = df[
        (df["ego_label"] == "moderate_large_circles")
        & (df["severity_regime"] == "high")
        & (
            df["method"].isin(
                [
                    "top_degree_blocking",
                    "greedy_blocking",
                    "harm_aware_greedy_blocking",
                    "harm_aware_resilience_greedy_blocking",
                ]
            )
        )
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in sorted(sub["method"].unique()):
        m = sub[sub["method"] == method].sort_values("budget_requested")
        ax.plot(m["budget_requested"], m["mean_total_harm"], marker="o", label=method)
    ax.set_xlabel("Requested blockers")
    ax.set_ylabel("Mean total harm")
    ax.set_title("Budget vs total harm (moderate_large_circles, high severity)")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_budget_vs_total_harm.png", dpi=150)
    plt.close(fig)


def figure_binary_vs_harm_methods(df: pd.DataFrame) -> None:
    sub = df[(df["budget_requested"] == 10) & (df["severity_regime"] == "medium")]
    g = (
        sub.groupby("method", as_index=False)
        .agg(
            mean_total_harm=("mean_total_harm", "mean"),
            se_total_harm=("mean_total_harm", lambda s: float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else 0.0),
        )
        .sort_values("mean_total_harm")
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(g))
    ax.barh(y, g["mean_total_harm"], xerr=g["se_total_harm"], capsize=3, alpha=0.88)
    ax.set_yticks(y)
    ax.set_yticklabels(g["method"])
    ax.set_xlabel("Mean total harm (+/- SE across ego cells)")
    ax.set_title("Binary vs harm-aware methods (k=10, medium severity)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_binary_vs_harm_method_comparison.png", dpi=150)
    plt.close(fig)


def figure_severity_propagation_robustness(df: pd.DataFrame) -> None:
    methods = ["greedy_blocking", "harm_aware_greedy_blocking"]
    order = ["low", "medium", "high"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for method in methods:
        for propagation_mode in ["constant", "bucket", "linear"]:
            sub = df[(df["method"] == method) & (df["propagation_mode"] == propagation_mode)].copy()
            g = sub.groupby("severity_regime", as_index=False).agg(mean_total_harm=("mean_total_harm", "mean"))
            g["severity_regime"] = pd.Categorical(g["severity_regime"], categories=order, ordered=True)
            g = g.sort_values("severity_regime")
            label = f"{method} | {propagation_mode}"
            ax.plot(np.arange(len(g)), g["mean_total_harm"], marker="o", label=label)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_xlabel("Severity regime")
    ax.set_ylabel("Mean total harm")
    ax.set_title("Severity-dependent propagation robustness")
    ax.legend(fontsize=6, loc="best")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_severity_propagation_robustness.png", dpi=150)
    plt.close(fig)


def figure_worst_community_harm_tradeoff(df: pd.DataFrame) -> None:
    methods = [
        "greedy_blocking",
        "fairness_aware_greedy_blocking",
        "harm_aware_greedy_blocking",
        "harm_aware_resilience_greedy_blocking",
    ]
    sub = df[
        (df["budget_requested"] == 10)
        & (df["severity_regime"] == "high")
        & (df["method"].isin(methods))
    ]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for method in methods:
        m = sub[sub["method"] == method]
        ax.scatter(
            m["mean_total_harm"],
            m["mean_worst_community_harm_rate"],
            label=method,
            alpha=0.8,
            s=45,
        )
    ax.set_xlabel("Mean total harm")
    ax.set_ylabel("Mean worst-community harm rate")
    ax.set_title("Harm efficiency vs concentrated structural harm")
    ax.legend(fontsize=6, loc="best")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_worst_community_harm_tradeoff.png", dpi=150)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TBL_DIR.mkdir(parents=True, exist_ok=True)

    method_df = _load_csv("harm_method_comparison_results.csv")
    sensitivity_df = _load_csv("severity_sensitivity_results.csv")

    build_table_binary_vs_harm_summary(method_df)
    build_table_severity_regime_summary()
    build_table_harm_budget_shortfall(method_df)
    build_table_harm_case_study(method_df)
    build_table_harm_robustness(sensitivity_df)

    figure_budget_vs_infected(method_df)
    figure_budget_vs_total_harm(method_df)
    figure_binary_vs_harm_methods(method_df)
    figure_severity_propagation_robustness(sensitivity_df)
    figure_worst_community_harm_tradeoff(method_df)

    logger.info("Wrote harm figures and tables")


if __name__ == "__main__":
    main()
