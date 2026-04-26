"""
Build unified CoAID article and engagement tables from News* _5 / _7 splits.

Primary key: (label, article_id) — fake and real pools reuse numeric article_id ranges.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

from src.config import (
    COAID_ARTICLES_CSV,
    COAID_ARTICLES_ENGAGEMENT_CSV,
    COAID_ENGAGEMENT_CSV,
    COAID_RAW,
    OUTPUTS_TABLES,
)

logger = logging.getLogger(__name__)

ARTICLE_COLS = [
    "article_id",
    "label",
    "type",
    "fact_check_url",
    "news_url",
    "title",
    "newstitle",
    "content",
    "abstract",
    "publish_date",
    "meta_keywords",
]


def _rename_first_column_to_article_id(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the leading index column (possibly unnamed or BOM-prefixed) to ``article_id``."""
    first = df.columns[0]
    out = df.rename(columns={first: "article_id"})
    out["article_id"] = pd.to_numeric(out["article_id"], errors="coerce")
    if out["article_id"].isna().any():
        bad = int(out["article_id"].isna().sum())
        raise ValueError(f"Non-numeric article_id rows after coercion: {bad}")
    out["article_id"] = out["article_id"].astype(int)
    return out


def _read_news_fake(path: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = _rename_first_column_to_article_id(df)
    df["label"] = 1
    df["source_split"] = split
    keep = [c for c in ARTICLE_COLS if c != "label"] + ["label", "source_split"]
    missing = [c for c in ARTICLE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in {path}: {missing}")
    return df[keep]


def _read_news_real(path: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = _rename_first_column_to_article_id(df)
    df["label"] = 0
    df["source_split"] = split
    missing = [c for c in ARTICLE_COLS if c not in df.columns and c != "label"]
    if missing:
        raise ValueError(f"Missing expected columns in {path}: {missing}")
    return df[[c for c in ARTICLE_COLS if c in df.columns] + ["source_split"]]


def build_fake_articles() -> pd.DataFrame:
    """Union fake _5 and _7 (disjoint article_id ranges per Phase 1 audit)."""
    a = _read_news_fake(COAID_RAW / "NewsFakeCOVID-19_5.csv", "5")
    b = _read_news_fake(COAID_RAW / "NewsFakeCOVID-19_7.csv", "7")
    out = pd.concat([a, b], ignore_index=True)
    dup = out.duplicated(subset=["label", "article_id"], keep=False)
    if dup.any():
        raise ValueError(
            f"Unexpected duplicate (label, article_id) in fake union: {out.loc[dup, ['article_id']].head()}"
        )
    return out


def build_real_articles() -> Tuple[pd.DataFrame, int]:
    """
    Concatenate real _5 and _7 and deduplicate by ``article_id``.

    **Rule:** rows from ``_5`` are kept first; a row from ``_7`` with the same
    ``article_id`` is dropped. ``source_split`` reflects the surviving row (always
    ``5`` when both splits contained the id).
    """
    a = _read_news_real(COAID_RAW / "NewsRealCOVID-19_5.csv", "5")
    b = _read_news_real(COAID_RAW / "NewsRealCOVID-19_7.csv", "7")
    stacked = pd.concat([a, b], ignore_index=True)
    n_before = len(stacked)
    out = stacked.drop_duplicates(subset=["article_id"], keep="first").copy()
    n_dup = n_before - len(out)
    return out, n_dup


def build_articles_table() -> Tuple[pd.DataFrame, int]:
    """Return unified articles and number of real duplicates dropped."""
    fake = build_fake_articles()
    real, n_real_dup = build_real_articles()
    articles = pd.concat([fake, real], ignore_index=True)
    dup = articles.duplicated(subset=["label", "article_id"], keep=False)
    if dup.any():
        raise ValueError("Duplicate (label, article_id) in final articles table.")
    return articles, n_real_dup


def _read_tweets(label: int, split: str) -> pd.DataFrame:
    kind = "Fake" if label == 1 else "Real"
    path = COAID_RAW / f"News{kind}COVID-19_tweets_{split}.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "index" not in df.columns or "tweet_id" not in df.columns:
        raise ValueError(f"Unexpected tweet schema in {path}: {list(df.columns)}")
    df = df.rename(columns={"index": "article_id"})
    df["article_id"] = df["article_id"].astype(int)
    df["label"] = label
    df["tweet_split"] = split
    return df[["label", "article_id", "tweet_id", "tweet_split"]]


def _read_replies(label: int, split: str) -> pd.DataFrame:
    kind = "Fake" if label == 1 else "Real"
    path = COAID_RAW / f"News{kind}COVID-19_tweets_replies_{split}.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    if not {"news_id", "tweet_id", "reply_id"}.issubset(df.columns):
        raise ValueError(f"Unexpected reply schema in {path}: {list(df.columns)}")
    df = df.rename(columns={"news_id": "article_id"})
    df["article_id"] = df["article_id"].astype(int)
    df["label"] = label
    df["reply_split"] = split
    return df[["label", "article_id", "tweet_id", "reply_id", "reply_split"]]


def build_engagement_frame() -> pd.DataFrame:
    """Aggregate tweet and reply tables across _5 and _7 for both labels."""
    tweets = pd.concat(
        [
            _read_tweets(1, "5"),
            _read_tweets(1, "7"),
            _read_tweets(0, "5"),
            _read_tweets(0, "7"),
        ],
        ignore_index=True,
    )
    replies = pd.concat(
        [
            _read_replies(1, "5"),
            _read_replies(1, "7"),
            _read_replies(0, "5"),
            _read_replies(0, "7"),
        ],
        ignore_index=True,
    )

    tw = (
        tweets.groupby(["label", "article_id"], as_index=False)
        .agg(
            n_related_tweets=("tweet_id", "count"),
            n_unique_tweets=("tweet_id", "nunique"),
        )
    )
    rp = (
        replies.groupby(["label", "article_id"], as_index=False)
        .agg(
            n_replies=("reply_id", "count"),
            n_unique_reply_tweets=("tweet_id", "nunique"),
        )
    )
    eng = tw.merge(rp, on=["label", "article_id"], how="outer")
    eng["n_related_tweets"] = eng["n_related_tweets"].fillna(0).astype(int)
    eng["n_unique_tweets"] = eng["n_unique_tweets"].fillna(0).astype(int)
    eng["n_replies"] = eng["n_replies"].fillna(0).astype(int)
    eng["n_unique_reply_tweets"] = eng["n_unique_reply_tweets"].fillna(0).astype(int)
    eng["engagement_total"] = eng["n_related_tweets"] + eng["n_replies"]
    return eng


def _write_missingness(df: pd.DataFrame, path: Path) -> None:
    rows = []
    for col in df.columns:
        series = df[col]
        empty = series.isna() | (series.astype(str).str.strip() == "")
        rows.append({"column": col, "pct_empty": round(100.0 * float(empty.mean()), 4)})
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    COAID_ARTICLES_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUTPUTS_TABLES.mkdir(parents=True, exist_ok=True)

    articles, n_real_dup = build_articles_table()
    engagement = build_engagement_frame()

    merged = articles.merge(engagement, on=["label", "article_id"], how="left")
    zcols = [
        "n_related_tweets",
        "n_unique_tweets",
        "n_replies",
        "n_unique_reply_tweets",
        "engagement_total",
    ]
    for c in zcols:
        merged[c] = merged[c].fillna(0)
        if c != "engagement_total":
            merged[c] = merged[c].astype(int)
        else:
            merged[c] = merged[c].astype(int)

    articles.to_csv(COAID_ARTICLES_CSV, index=False)
    engagement.to_csv(COAID_ENGAGEMENT_CSV, index=False)
    merged.to_csv(COAID_ARTICLES_ENGAGEMENT_CSV, index=False)

    summary = pd.DataFrame(
        [
            {
                "n_fake": int((articles["label"] == 1).sum()),
                "n_real": int((articles["label"] == 0).sum()),
                "n_total": len(articles),
                "n_real_duplicates_dropped": n_real_dup,
                "n_engagement_rows": len(engagement),
            }
        ]
    )
    summary.to_csv(OUTPUTS_TABLES / "coaid_summary.csv", index=False)
    _write_missingness(merged, OUTPUTS_TABLES / "coaid_missingness.csv")

    logger.info("Wrote %s (%d rows)", COAID_ARTICLES_CSV, len(articles))
    logger.info("Wrote %s (%d rows)", COAID_ENGAGEMENT_CSV, len(engagement))
    logger.info("Wrote %s (%d rows)", COAID_ARTICLES_ENGAGEMENT_CSV, len(merged))


if __name__ == "__main__":
    main()
