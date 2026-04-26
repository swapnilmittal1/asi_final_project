"""Project paths and defaults. Expand as the pipeline grows."""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_RAW: Path = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM: Path = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED: Path = PROJECT_ROOT / "data" / "processed"
OUTPUTS: Path = PROJECT_ROOT / "outputs"
OUTPUTS_TABLES: Path = OUTPUTS / "tables"

COAID_RAW: Path = DATA_RAW / "coaid"
TWITTER_RAW: Path = DATA_RAW / "twitter"

COAID_ARTICLES_CSV: Path = DATA_PROCESSED / "coaid_articles.csv"
COAID_ENGAGEMENT_CSV: Path = DATA_PROCESSED / "coaid_engagement.csv"
COAID_ARTICLES_ENGAGEMENT_CSV: Path = DATA_PROCESSED / "coaid_articles_with_engagement.csv"
COAID_ARTICLE_SEVERITY_CSV: Path = DATA_PROCESSED / "coaid_article_severity.csv"
COAID_SEVERITY_REGIMES_CSV: Path = DATA_PROCESSED / "coaid_severity_regimes.csv"
SCENARIOS_DIR: Path = DATA_PROCESSED / "scenarios"
HARM_SCENARIOS_DIR: Path = DATA_PROCESSED / "harm_scenarios"
SELECTED_EGOS_JSON: Path = DATA_PROCESSED / "selected_egos.json"
EGOS_PROCESSED_DIR: Path = DATA_PROCESSED / "egos"
