"""
pipeline.common
---------------
Shared paths, constants and load/save helpers for the
"Turbulence in Words" text-mining pipeline.

Every module in the pipeline package uses these helpers so that
artifact locations stay consistent and the pipeline is reproducible.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

os.environ.setdefault("NLTK_ALLOW_PROXIED_URLOPEN", "1")
try:
    import nltk
    nltk.pathsec.ALLOW_PROXIED_FETCH = True
except Exception:
    pass

import pandas as pd

# --------------------------------------------------------------------------
# Project-level constants
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("turbulence")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"
REPORT_DIR = PROJECT_ROOT / "report"

RAW_CSV = DATA_DIR / "raw_airline_reviews.csv"
CLEAN_CSV = DATA_DIR / "clean_reviews.csv"
CLEANING_STATS_JSON = DATA_DIR / "cleaning_stats.json"

RANDOM_STATE = 42

# Skytrax per-aspect rating columns (1-5 stars, "n" == not rated)
ASPECT_RATING_COLS = [
    "seat_comfort_rating",
    "cabin_staff_rating",
    "food_beverages_rating",
    "inflight_entertainment_rating",
    "ground_service_rating",
    "wifi_connectivity_rating",
    "value_money_rating",
]

# Raw source dataset (public GitHub mirror, no API key required)
SOURCE_URL = (
    "https://raw.githubusercontent.com/quankiquanki/"
    "skytrax-reviews-dataset/master/data/airline.csv"
)

# Phase 2 feature artifacts
DTM_PKL = FEATURES_DIR / "dtm.pkl"
ENGINEERED_CSV = FEATURES_DIR / "engineered.csv"
FEATURES_META_JSON = FEATURES_DIR / "features_meta.json"

# Phase 2 - dimensionality reduction
LSA_PKL = FEATURES_DIR / "lsa.pkl"
LDA_MODEL_PKL = MODELS_DIR / "lda_model.pkl"
TSNE_CSV = DATA_DIR / "tsne_coords.csv"
TOPIC_TERMS_JSON = DATA_DIR / "topic_terms.json"
TOPIC_MIX_CSV = DATA_DIR / "topic_mix.csv"
LDA_PERPLEXITY_JSON = DATA_DIR / "lda_perplexity.json"

# Phase 3 artifacts - classification
BEST_CLASSIFIER_PKL = MODELS_DIR / "best_classifier.pkl"
METRICS_JSON = MODELS_DIR / "metrics.json"
LIME_SAMPLES_JSON = MODELS_DIR / "lime_samples.json"
ERRORS_CSV = MODELS_DIR / "errors.csv"
ROC_CURVES_CSV = MODELS_DIR / "roc_curves.csv"
CLASSIFIER_TRAINTEST_CSV = MODELS_DIR / "test_predictions.csv"

# Phase 3 artifacts - tagging
TAGGER_LEXICONS_JSON = MODELS_DIR / "tagger_lexicons.json"
ASPECT_TAGS_CSV = DATA_DIR / "aspect_tags.csv"
TAGGER_VALIDATION_JSON = DATA_DIR / "tagger_validation.json"
CONFLICTS_CSV = DATA_DIR / "conflicts.csv"
SCORECARD_CSV = DATA_DIR / "scorecard.csv"

# Phase 4 insights
TRENDS_CSV = DATA_DIR / "trends.csv"
AIRLINE_CLUSTERS_CSV = DATA_DIR / "airline_clusters.csv"
PSI_RANKING_CSV = DATA_DIR / "psi_ranking.csv"
NPS_CSV = DATA_DIR / "nps.csv"
ASPECT_NETWORK_JSON = DATA_DIR / "aspect_network.json"
CABIN_COUNTRY_CSV = DATA_DIR / "cabin_country.csv"
WORD2VEC_PKL = MODELS_DIR / "word2vec.pkl"
LDA_MODEL_PKL = MODELS_DIR / "lda_model.pkl"
TSNE_CSV = DATA_DIR / "tsne_coords.csv"
TOPIC_TERMS_JSON = DATA_DIR / "topic_terms.json"
TOPIC_MIX_CSV = DATA_DIR / "topic_mix.csv"
LDA_PERPLEXITY_JSON = DATA_DIR / "lda_perplexity.json"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def ensure_dirs() -> None:
    """Create all artifact directories if they do not exist yet."""
    for d in (DATA_DIR, FEATURES_DIR, MODELS_DIR, FIGURES_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, path: Path, **kwargs) -> Path:
    """Save a DataFrame as CSV, creating parent directories as needed."""
    path = Path(path)
    ensure_dirs()
    df.to_csv(path, index=kwargs.pop("index", False), **kwargs)
    logger.info("Saved %s (%s rows x %s cols)", path.name, len(df), df.shape[1])
    return path


def load_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Load a CSV artifact."""
    return pd.read_csv(path, **kwargs)


def save_json(obj, path: Path) -> Path:
    """Save any JSON-serialisable object."""
    path = Path(path)
    ensure_dirs()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
    logger.info("Saved %s", path.name)
    return path


def load_json(path: Path):
    """Load a JSON artifact."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
