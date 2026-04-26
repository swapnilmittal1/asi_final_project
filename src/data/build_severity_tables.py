"""
Build severity tables from CoAID article metadata.

Severity is heuristic and scenario-level by design: CoAID provides article text
and engagement signals, but there is no user-level linkage from articles to SNAP
nodes. The resulting tables are therefore used to parameterize semi-synthetic
severity regimes, not to claim ground-truth real-world harm labels.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from src.config import (
    COAID_ARTICLE_SEVERITY_CSV,
    COAID_ARTICLES_ENGAGEMENT_CSV,
    COAID_SEVERITY_REGIMES_CSV,
    DATA_PROCESSED,
    OUTPUTS_TABLES,
)
from src.models.severity import (
    DEFAULT_SEVERITY_RUBRIC,
    SeverityProfile,
    make_profile_from_score,
    propagation_multiplier_from_score,
)

logger = logging.getLogger(__name__)

MANUAL_OVERRIDE_CSV = DATA_PROCESSED / "coaid_severity_manual_overrides.csv"

HIGH_RISK_TERMS: Tuple[str, ...] = (
    "vaccine",
    "vaccination",
    "inject",
    "bleach",
    "disinfectant",
    "drink",
    "garlic",
    "cure",
    "treatment",
    "hydroxychloroquine",
    "ivermectin",
    "mask",
    "hospital alone",
    "refuse vaccination",
)

MEDIUM_RISK_TERMS: Tuple[str, ...] = (
    "testing",
    "test kit",
    "symptom",
    "infection",
    "cases",
    "death",
    "lockdown",
    "quarantine",
    "hospital",
    "immunity",
    "spread",
    "origin",
)

GUIDANCE_TERMS: Tuple[str, ...] = (
    "should",
    "must",
    "need to",
    "avoid",
    "take",
    "stop",
    "refuse",
    "prevent",
    "proof",
)


def _normalize_text(parts: Iterable[Any]) -> str:
    cleaned: List[str] = []
    for value in parts:
        if value is None:
            continue
        text = str(value).strip().lower()
        if not text or text == "nan":
            continue
        cleaned.append(text)
    return " ".join(cleaned)


def _keyword_hits(text: str, terms: Iterable[str]) -> int:
    return sum(1 for term in terms if term in text)


def _fake_engagement_quantiles(df: pd.DataFrame) -> Tuple[float, float]:
    fake = df[df["label"] == 1]["engagement_total"].astype(float)
    if fake.empty:
        return 0.0, 0.0
    return tuple(np.quantile(fake, [1.0 / 3.0, 2.0 / 3.0]).tolist())  # type: ignore[return-value]


def _engagement_bucket(value: float, cutoffs: Tuple[float, float]) -> str:
    low_cut, high_cut = cutoffs
    if value <= low_cut:
        return "low"
    if value <= high_cut:
        return "medium"
    return "high"


def _score_fake_article(
    row: pd.Series,
    *,
    engagement_cutoffs: Tuple[float, float],
) -> Dict[str, Any]:
    text = _normalize_text(
        [
            row.get("title"),
            row.get("newstitle"),
            row.get("abstract"),
            row.get("content"),
            row.get("meta_keywords"),
            row.get("fact_check_url"),
        ]
    )
    high_hits = _keyword_hits(text, HIGH_RISK_TERMS)
    medium_hits = _keyword_hits(text, MEDIUM_RISK_TERMS)
    guidance_hits = _keyword_hits(text, GUIDANCE_TERMS)

    engagement_total = float(row.get("engagement_total", 0.0))
    engagement_bucket = _engagement_bucket(engagement_total, engagement_cutoffs)
    engagement_bonus = {"low": 0.0, "medium": 0.05, "high": 0.1}[engagement_bucket]

    # Content dominates; engagement only slightly adjusts plausible harm exposure.
    score = 0.15
    score += min(0.45, 0.18 * float(high_hits))
    score += min(0.20, 0.08 * float(medium_hits))
    score += min(0.15, 0.06 * float(guidance_hits))
    if high_hits > 0 and guidance_hits > 0:
        score += 0.10
    score += engagement_bonus
    score = max(0.0, min(1.0, score))

    profile = make_profile_from_score(
        score,
        construction_source="coaid_rule_based",
        propagation_mode="bucket",
        notes="Heuristic severity from content/guidance keywords with mild engagement adjustment.",
    )
    return {
        "severity_applicable": True,
        "severity_score": round(profile.severity_score, 4),
        "severity_label": profile.severity_label,
        "base_harm_weight": profile.base_harm_weight,
        "propagation_multiplier_constant": 1.0,
        "propagation_multiplier_bucket": round(profile.propagation_multiplier, 4),
        "propagation_multiplier_linear": round(
            propagation_multiplier_from_score(profile.severity_score, mode="linear"),
            4,
        ),
        "severity_strategy": "rule_based_default",
        "severity_notes": profile.notes,
        "engagement_bucket_fake_only": engagement_bucket,
        "high_risk_keyword_hits": high_hits,
        "medium_risk_keyword_hits": medium_hits,
        "guidance_keyword_hits": guidance_hits,
    }


def _default_non_fake_assignment() -> Dict[str, Any]:
    return {
        "severity_applicable": False,
        "severity_score": 0.0,
        "severity_label": "not_applicable",
        "base_harm_weight": 0.0,
        "propagation_multiplier_constant": 1.0,
        "propagation_multiplier_bucket": 1.0,
        "propagation_multiplier_linear": 1.0,
        "severity_strategy": "not_fake_label",
        "severity_notes": "Severity rubric is only applied to fake-labeled CoAID rows.",
        "engagement_bucket_fake_only": "not_applicable",
        "high_risk_keyword_hits": 0,
        "medium_risk_keyword_hits": 0,
        "guidance_keyword_hits": 0,
    }


def _apply_manual_overrides(df: pd.DataFrame) -> pd.DataFrame:
    if not MANUAL_OVERRIDE_CSV.exists():
        return df
    overrides = pd.read_csv(MANUAL_OVERRIDE_CSV)
    required = {"label", "article_id", "severity_score"}
    missing = required.difference(overrides.columns)
    if missing:
        raise ValueError(f"Manual severity override CSV missing columns: {sorted(missing)}")
    overrides = overrides.copy()
    overrides["label"] = overrides["label"].astype(int)
    overrides["article_id"] = overrides["article_id"].astype(int)
    merged = df.merge(
        overrides,
        on=["label", "article_id"],
        how="left",
        suffixes=("", "_override"),
    )
    has_override = merged["severity_score_override"].notna()
    if not has_override.any():
        return df
    merged.loc[has_override, "severity_score"] = merged.loc[has_override, "severity_score_override"]
    merged.loc[has_override, "severity_label"] = merged.loc[has_override, "severity_label_override"].fillna(
        merged.loc[has_override, "severity_label"]
    )
    merged.loc[has_override, "base_harm_weight"] = merged.loc[
        has_override, "base_harm_weight_override"
    ].fillna(merged.loc[has_override, "base_harm_weight"])
    merged.loc[has_override, "severity_strategy"] = "manual_override"
    if "notes_override" in merged.columns:
        merged.loc[has_override, "severity_notes"] = merged.loc[has_override, "notes_override"].fillna(
            merged.loc[has_override, "severity_notes"]
        )
    drop_cols = [c for c in merged.columns if c.endswith("_override")]
    return merged.drop(columns=drop_cols)


def build_article_severity_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return article-level severity assignments for all CoAID rows.
    """
    engagement_cutoffs = _fake_engagement_quantiles(df)
    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        base = row.to_dict()
        if int(row["label"]) == 1:
            base.update(_score_fake_article(row, engagement_cutoffs=engagement_cutoffs))
        else:
            base.update(_default_non_fake_assignment())
        records.append(base)
    out = pd.DataFrame(records)
    out = _apply_manual_overrides(out)
    return out


def build_severity_regime_table(article_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize fake-article severity buckets into reusable scenario regimes.
    """
    fake = article_df[article_df["severity_applicable"] == True].copy()  # noqa: E712
    rows: List[Dict[str, Any]] = []
    for regime in ["low", "medium", "high"]:
        sub = fake[fake["severity_label"] == regime].copy()
        if sub.empty:
            continue
        median_score = float(sub["severity_score"].median())
        profile = make_profile_from_score(
            median_score,
            construction_source="coaid_regime_median",
            propagation_mode="bucket",
            notes="Median scenario profile for the severity bucket.",
        )
        rows.append(
            {
                "severity_regime": regime,
                "n_articles": int(len(sub)),
                "severity_score_mean": round(float(sub["severity_score"].mean()), 4),
                "severity_score_median": round(median_score, 4),
                "base_harm_weight": profile.base_harm_weight,
                "propagation_multiplier_constant": 1.0,
                "propagation_multiplier_bucket": round(profile.propagation_multiplier, 4),
                "propagation_multiplier_linear": round(
                    propagation_multiplier_from_score(median_score, mode="linear"),
                    4,
                ),
                "construction_source": profile.construction_source,
                "severity_notes": (
                    "Bucket summary over fake-labeled CoAID articles. Use as semi-synthetic "
                    "scenario metadata, not as a real user-level harm label."
                ),
            }
        )
    return pd.DataFrame(rows)


def build_llm_review_stub(article_df: pd.DataFrame) -> pd.DataFrame:
    """
    Export ambiguous fake rows for optional manual or LLM-assisted review.

    No API call is made here; this file is only a review queue.
    """
    fake = article_df[article_df["severity_applicable"] == True].copy()  # noqa: E712
    fake["distance_to_threshold"] = fake["severity_score"].apply(
        lambda x: min(
            abs(float(x) - DEFAULT_SEVERITY_RUBRIC.low_max),
            abs(float(x) - DEFAULT_SEVERITY_RUBRIC.medium_max),
        )
    )
    cols = [
        "label",
        "article_id",
        "title",
        "abstract",
        "content",
        "severity_score",
        "severity_label",
        "distance_to_threshold",
        "severity_strategy",
    ]
    out = fake.sort_values("distance_to_threshold").head(50)[cols].copy()
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not COAID_ARTICLES_ENGAGEMENT_CSV.exists():
        raise FileNotFoundError(
            f"Missing {COAID_ARTICLES_ENGAGEMENT_CSV}; run python -m src.data.build_coaid_tables first."
        )

    OUTPUTS_TABLES.mkdir(parents=True, exist_ok=True)
    COAID_ARTICLE_SEVERITY_CSV.parent.mkdir(parents=True, exist_ok=True)

    articles = pd.read_csv(COAID_ARTICLES_ENGAGEMENT_CSV)
    article_severity = build_article_severity_table(articles)
    severity_regimes = build_severity_regime_table(article_severity)
    llm_stub = build_llm_review_stub(article_severity)

    article_severity.to_csv(COAID_ARTICLE_SEVERITY_CSV, index=False)
    severity_regimes.to_csv(COAID_SEVERITY_REGIMES_CSV, index=False)
    llm_stub.to_csv(OUTPUTS_TABLES / "coaid_severity_review_stub.csv", index=False)

    summary_rows: List[Dict[str, Any]] = []
    fake = article_severity[article_severity["severity_applicable"] == True]  # noqa: E712
    for regime in ["low", "medium", "high"]:
        sub = fake[fake["severity_label"] == regime]
        summary_rows.append(
            {
                "severity_regime": regime,
                "n_articles": int(len(sub)),
                "mean_score": round(float(sub["severity_score"].mean()), 4) if not sub.empty else 0.0,
                "mean_harm_weight": round(float(sub["base_harm_weight"].mean()), 4) if not sub.empty else 0.0,
            }
        )
    pd.DataFrame(summary_rows).to_csv(OUTPUTS_TABLES / "coaid_severity_summary.csv", index=False)

    rubric_stub = {
        "high_risk_terms": list(HIGH_RISK_TERMS),
        "medium_risk_terms": list(MEDIUM_RISK_TERMS),
        "guidance_terms": list(GUIDANCE_TERMS),
        "manual_override_csv": str(MANUAL_OVERRIDE_CSV),
    }
    (OUTPUTS_TABLES / "coaid_severity_rubric_stub.json").write_text(
        json.dumps(rubric_stub, indent=2),
        encoding="utf-8",
    )

    logger.info("Wrote %s (%d rows)", COAID_ARTICLE_SEVERITY_CSV, len(article_severity))
    logger.info("Wrote %s (%d rows)", COAID_SEVERITY_REGIMES_CSV, len(severity_regimes))
    logger.info("Wrote %s", OUTPUTS_TABLES / "coaid_severity_summary.csv")


if __name__ == "__main__":
    main()
